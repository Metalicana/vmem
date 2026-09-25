import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from scripts import run_vmem_results as workflow
from scripts import vmem_slurm as slurm
from scripts.setup_vmem_newton import runtime_requirements
from scripts.profile_vmem_newton import action_profile


class SlurmAdmissionTest(unittest.TestCase):
    def test_preserves_scheduler_mask_not_physical_index(self):
        for mask in ("0", "3", "GPU-allocated", "MIG-allocated"):
            env = {"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": mask, "PATH": "/bin"}
            result = slurm.child_environment("inherit", True, env)
            self.assertEqual(result, env)
            self.assertEqual(slurm.child_environment("inherit", False, env)["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], mask)

    def test_inherit_requires_allocation_and_mask(self):
        for env in ({}, {"CUDA_VISIBLE_DEVICES": "1"}, {"SLURM_JOB_ID": "42"},
                    {"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": "-1"}):
            with self.assertRaises(ValueError):
                slurm.child_environment("inherit", True, env)

    def test_slurm_refuses_detaching_and_physical_selection(self):
        env = {"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": "2"}
        for action, gpu in (("start", "inherit"), ("start", "1"), ("run", "0"), ("run", "GPU-abc")):
            with self.assertRaisesRegex(ValueError, "foreground"):
                slurm.check_launch(action, gpu, env)
        slurm.check_launch("run", "inherit", env)
        slurm.check_launch("start", "1", {})
        slurm.check_launch("status", "inherit", {})

    def test_workstation_environment_unchanged(self):
        self.assertEqual(slurm.child_environment("1", True, {})["CUDA_VISIBLE_DEVICES"], "1")

    def test_capacity_fails_impossible_and_insufficient_requests(self):
        snapshot = {"free_mib": 78000, "total_mib": 81000}
        with self.assertRaisesRegex(ValueError, "capacity"):
            slurm.check_capacity(snapshot, 85000)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            slurm.check_capacity(snapshot, 79000)
        slurm.check_capacity(snapshot, 70000)

    def test_allocated_probe_uses_cuda_not_nvml_index(self):
        with patch.dict(slurm.os.environ, {"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": "7"}), \
                patch.object(slurm.subprocess, "run", return_value=SimpleNamespace(stdout='{"free_mib":78000}')) as run:
            self.assertEqual(slurm.query_allocated_gpu()["free_mib"], 78000)
        self.assertIn("vmem_slurm.py", run.call_args.args[0][1])
        self.assertNotIn("nvidia-smi", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 180)

    def test_allocated_wait_does_not_poll_other_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), generation_root=Path(tmp), gpu="inherit", min_free_mib=70000)
            instance = workflow.Workflow(args)
            with patch("vmem_slurm.query_allocated_gpu", return_value={"free_mib": 78000, "total_mib": 81000}), \
                    patch.object(workflow, "query_gpu", side_effect=AssertionError("physical query")), \
                    patch.object(workflow.time, "sleep", side_effect=AssertionError("idle wait")):
                instance.wait_gpu("generation")

    def test_gpu_child_has_same_process_cuda_guard_and_exact_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = workflow.Workflow(SimpleNamespace(output=Path(tmp), generation_root=Path(tmp), gpu="inherit"))
            child = MagicMock(pid=999, returncode=0)
            child.wait.return_value = 0
            child.poll.return_value = 0
            with patch.dict(workflow.os.environ, {"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": "7"}), \
                    patch.object(instance, "wait_gpu"), patch.object(workflow.subprocess, "Popen", return_value=child) as popen:
                instance.command("generation", ["/env/vmem/bin/python", "/repo/scripts/run_vmem_demo_actions.py"], gpu=True)
            command = popen.call_args.args[0]
            self.assertEqual(Path(command[1]).name, "vmem_slurm.py")
            self.assertEqual(command[2], "--execute")
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "7")


class NewtonProtocolTest(unittest.TestCase):
    def test_smoke_is_same_oxford_prefix_through_eviction_and_return(self):
        from scripts.run_vmem_demo_manifest import _load_manifest
        from scripts.run_vmem_demo_actions import _expand_trajectory_actions
        from scripts.vmem_protocol import expected_settings, commanded_path
        root = Path(__file__).resolve().parents[1]
        smoke = _load_manifest(root / "manifests/vmem_newton_smoke_v1.jsonl")
        long = _load_manifest(root / "manifests/vmem_transfer_v3.jsonl")[:2]
        self.assertEqual(len(smoke), 2)
        for left, right in zip(smoke, long):
            a, b = expected_settings(left), expected_settings(right)
            ignore = {"run_id", "num_actions"}
            self.assertEqual({k: v for k, v in a.items() if k not in ignore},
                             {k: v for k, v in b.items() if k not in ignore})
            actions = _expand_trajectory_actions(SimpleNamespace(**a))
            self.assertEqual(actions, _expand_trajectory_actions(SimpleNamespace(**b))[:18])
            self.assertEqual(1 + len(actions) * 4, 73)
            self.assertTrue(commanded_path(actions, .1)[-1]["at_origin"])
        self.assertEqual(smoke[1]["memory_budget"], 32)

    def test_install_split_keeps_numpy_and_requires_review_on_requirements_change(self):
        requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
        result = runtime_requirements(requirements)
        self.assertIn("numpy==1.24.4", result)
        self.assertNotIn("torch==2.7.0", result)
        self.assertNotIn("nightly", result)
        self.assertNotIn("curope", result)
        with self.assertRaises(ValueError):
            runtime_requirements(requirements.replace("torch==2.7.0", "torch==2.7.1"))
        with self.assertRaises(ValueError):
            runtime_requirements(requirements + "\n--extra-index-url https://example.invalid\n")

    def test_profile_warmup_and_projection_scope(self):
        actions = [{"action_index": i, "wall_seconds": value} for i, value in enumerate((100, 80, 10, 20, 30))]
        steps = [{"step": i, "warmup": i < 2} for i in range(5)]
        result = action_profile(actions, steps)
        self.assertEqual(result["median_action_seconds_excluding_warmup"], 20)
        self.assertEqual(result["first_action_seconds"], 100)
        self.assertAlmostEqual(result["constant_late_rate_195_actions_hours"], 20 * 195 / 3600)
        self.assertIn("NOT an ETA", result["timing_scope"])
        self.assertIsNone(action_profile([], [])["last_three_median_action_seconds"])

    def test_generation_only_does_not_require_quality_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = workflow.Workflow(SimpleNamespace(output=Path(tmp), generation_root=Path(tmp), gpu="inherit", generation_only=True))
            instance.rows = [{}, {}]
            instance.state["status"] = "complete"
            with patch("report_vmem_results.build_report", return_value={"errors": [], "quality": [], "pairing_checks": {"smoke": True}}):
                instance.report()
            self.assertEqual(instance.state["status"], "complete")

    def test_paused_smoke_report_has_no_invented_quality_or_sixty_second_label(self):
        from scripts.report_vmem_results import build_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow.save(root / "workflow_status.json", {"status": "paused", "errors": []})
            workflow.save(root / "workflow_spec.json", {"dimensions": [], "selected_rows": [{"num_actions": 18}]})
            result = build_report(root)
            self.assertEqual(result["quality"], [])
            self.assertFalse(result["quality_requested"])
            html = (root / "report/index.html").read_text()
            self.assertIn("18 actions", html)
            self.assertIn("quality evaluation not requested", html)
            self.assertNotIn("60 s /", html)
            self.assertTrue((root / "results.zip").is_file())

    def test_status_on_login_host_does_not_guess_remote_pid_liveness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow.save(root / "workflow_status.json", {"hostname": "compute-node", "slurm_job_id": "42",
                                                          "pid": 999, "process_token": "123"})
            with patch.object(workflow.socket, "gethostname", return_value="login-node"), \
                    patch.object(workflow, "process_token", side_effect=AssertionError("remote PID lookup")), \
                    patch("builtins.print") as output:
                workflow.status(SimpleNamespace(output=root, generation_root=root))
            self.assertIsNone(json.loads(output.call_args_list[0].args[0])["controller_alive"])


if __name__ == "__main__":
    unittest.main()
