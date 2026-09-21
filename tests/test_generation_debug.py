import ast
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from generation_debug import GenerationDebug, phase_seed, validate_debug_args
from scripts.audit_vmem_generation_debug import compare


ROOT = Path(__file__).resolve().parents[1]


class DebugContractTest(unittest.TestCase):
    def test_seeds_are_stable_and_phase_action_specific(self):
        self.assertEqual(phase_seed(501, "diffusion", 2), phase_seed(501, "diffusion", 2))
        values = {phase_seed(seed, phase, step) for seed in (501, 502)
                  for phase in ("initialization", "diffusion", "reconstruction") for step in (-1, 0, 1)}
        self.assertEqual(len(values), 18)
        self.assertTrue(all(0 <= value < 2 ** 63 for value in values))

    def test_debug_rejects_locked_resumed_long_or_checkpointed_runs(self):
        args = dict(generation_debug="observe", experiment_lock=None, resume_from=None,
                    checkpoint_every=0, frames_per_action=4, num_actions=2)
        validate_debug_args(SimpleNamespace(**args))
        for key, value in (("experiment_lock", "lock.json"), ("resume_from", "old-run"),
                           ("checkpoint_every", 5), ("num_actions", 13), ("frames_per_action", 8)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_debug_args(SimpleNamespace(**{**args, key: value}))
        validate_debug_args(SimpleNamespace(generation_debug=None))

    def test_modified_production_hooks_compile_without_importing_models(self):
        for relative in ("generation_debug.py", "utils/util.py", "modeling/pipeline.py",
                         "modeling/sampling.py", "scripts/run_vmem_demo_actions.py",
                         "scripts/audit_vmem_generation_debug.py"):
            with self.subTest(file=relative):
                compile((ROOT / relative).read_text(), relative, "exec")
        tree = ast.parse((ROOT / "utils/util.py").read_text())
        sampler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "do_sample")
        calls = [node for node in ast.walk(sampler) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and call.func.attr == "phase"
                            and call.args[0].value == "diffusion" for call in calls if call.args))


class DebugComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.left, self.right = [Path(self.tmp.name) / name for name in ("left", "right")]
        from scripts.vmem_protocol import expected_settings
        args = expected_settings({"image": "image.jpg", "run_id": "test", "num_actions": 1,
                                  "inference_steps": 2})
        args["generation_debug"] = "observe"
        self.rows = [{"event": "schema", "schema": "vmem_generation_debug_v1", "mode": "observe", "seed": args["seed"]}]
        for phase, step in (("model_load", -1), ("initialization", -1), ("diffusion", 0), ("reconstruction", 0)):
            self.rows.extend({"event": event, "phase": phase, "step": step}
                             for event in ("phase_start", "phase_end"))
        self.rows.extend({"event": event, "step": step} for event, step in (
            ("initial_input", -1), ("initial_encoding", -1), ("conditioning", 0),
            ("initial_noise", 0), ("diffusion_latents", 0), ("decoded_samples", 0)))
        self.rows.extend({"event": "sampler_noise", "step": 0, "sampler_step": index, "noise": "same"}
                         for index in range(2))
        for path in (self.left, self.right):
            path.mkdir()
            (path / "run_status.json").write_text(json.dumps({"status": "complete"}))
            (path / "run_spec.json").write_text(json.dumps({"arguments": args, "provenance": {
                key: "same" for key in ("image_sha256", "config_sha256", "source_sha256", "checkpoint_sha256")}}))
            (path / "generation_environment.json").write_text('{}')
            (path / "generation_debug.jsonl").write_text("\n".join(json.dumps(row) for row in self.rows))

    def write_rows(self):
        (self.right / "generation_debug.jsonl").write_text("\n".join(json.dumps(row) for row in self.rows))

    def test_same_policy_repeat_is_supported(self):
        report = compare(self.left, self.right)
        self.assertTrue(report["all_recorded_events_equal"])
        self.assertEqual(report["settings_mismatches"], [])
        self.assertEqual(report["missing_provenance"], [])

    def test_reports_first_sampler_noise_difference(self):
        self.rows[-1]["noise"] = "different"
        self.write_rows()
        report = compare(self.left, self.right)
        self.assertEqual(report["first_difference"]["event"], "sampler_noise")
        self.assertEqual(report["first_difference"]["sampler_step"], 1)

    def test_incomplete_noise_trace_is_rejected(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Sampler noise count"):
            compare(self.left, self.right)

    def test_failed_run_is_not_compared_as_complete(self):
        (self.right / "run_status.json").write_text('{"status":"failed"}')
        with self.assertRaisesRegex(ValueError, "not complete"):
            compare(self.left, self.right)


class GenerationDebugTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is required")
        cls.torch = torch

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(self.torch.set_rng_state, self.torch.get_rng_state())

    def debug(self, mode="observe", name="trace"):
        return GenerationDebug(self.root / f"{name}.jsonl", seed=501, mode=mode,
                               device="cpu", torch_module=self.torch)

    def reset(self):
        random.seed(123)
        np.random.seed(123)
        self.torch.set_rng_state(self.torch.Generator().manual_seed(123).get_state())

    def draws(self, count=10):
        return random.random(), np.random.random(count), self.torch.randn(count)

    def assert_draws_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertTrue(self.torch.equal(left[2], right[2]))

    def test_observe_preserves_draws_and_rng_consumption(self):
        self.reset()
        expected, expected_next = self.draws(), self.draws()
        self.reset()
        debug = self.debug()
        with debug.phase("diffusion", 0):
            actual = self.draws()
            debug.record("noise", 0, numpy=actual[1], torch=actual[2])
        self.assert_draws_equal(expected, actual)
        self.assert_draws_equal(expected_next, self.draws())

    def test_isolated_noise_ignores_background_and_reconstruction_draw_counts(self):
        debug = self.debug("isolated")
        with debug.phase("reconstruction", 0):
            self.draws(37 * 32)
        with debug.phase("diffusion", 1):
            first = self.draws()
        self.draws(100)
        with debug.phase("reconstruction", 0):
            self.draws(36 * 32)
        with debug.phase("diffusion", 1):
            second = self.draws()
        self.assert_draws_equal(first, second)

    def test_isolated_phase_restores_all_streams_on_failure(self):
        self.reset()
        expected = self.draws()
        self.reset()
        debug = self.debug("isolated")
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with debug.phase("diffusion", 0):
                self.draws(100)
                raise RuntimeError("failure")
        self.assert_draws_equal(expected, self.draws())
        rows = [json.loads(line) for line in debug.path.read_text().splitlines()]
        self.assertEqual(rows[-1]["event"], "phase_error")

    def test_fingerprints_preserve_values_layout_dtype_and_rng(self):
        debug = self.debug()
        tensor = self.torch.arange(12).reshape(3, 4).T
        before = tensor.clone()
        rng = self.torch.get_rng_state()
        self.assertEqual(debug.fingerprint(tensor), debug.fingerprint(tensor.contiguous()))
        self.assertNotEqual(debug.fingerprint(tensor), debug.fingerprint(tensor.float()))
        self.assertNotEqual(debug.fingerprint(tensor), debug.fingerprint(tensor + 1))
        self.assertEqual(len(debug.fingerprint(self.torch.tensor(1, dtype=self.torch.bfloat16))["sha256"]), 64)
        self.assertTrue(self.torch.equal(tensor, before))
        self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))

    def test_existing_trace_is_not_overwritten(self):
        debug = self.debug()
        before = debug.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.debug()
        self.assertEqual(debug.path.read_bytes(), before)

    def test_phase_identity_and_rng_fingerprints_are_recorded(self):
        debug = self.debug("isolated")
        with debug.phase("diffusion", 7):
            self.draws()
        rows = [json.loads(line) for line in debug.path.read_text().splitlines()]
        self.assertEqual(rows[1]["step"], 7)
        self.assertEqual(rows[1]["phase_seed"], phase_seed(501, "diffusion", 7))
        self.assertNotEqual(rows[1]["rng"]["cpu"], rows[2]["rng"]["cpu"])
        self.assertIsNone(rows[1]["rng"]["cuda"])

    def test_actual_sampler_callback_does_not_change_outputs_or_rng(self):
        spec = importlib.util.spec_from_file_location("debug_test_sampling", ROOT / "modeling/sampling.py")
        sampling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sampling)
        torch = self.torch

        class Guider:
            def prepare_inputs(self, x, sigma, cond, uc):
                return x, sigma, cond

            def __call__(self, prediction, sigma, scale, **kwargs):
                self_outer.assertEqual(kwargs, {})
                return prediction

        self_outer = self
        sampler = sampling.EulerEDMSampler(
            num_steps=2, discretization=lambda n, device: torch.tensor([1., .5, 0.], device=device),
            guider=Guider(), device="cpu", verbose=False)
        debug = self.debug()

        def sample(callback):
            self.reset()
            result = sampler(lambda x, sigma, cond: x * .5, torch.ones(1, 4, 2, 2),
                             scale=1., cond={}, verbose=False, debug_noise=callback)
            return result, torch.get_rng_state()

        expected, expected_rng = sample(None)
        actual, actual_rng = sample(lambda index, noise: debug.record("sampler_noise", 0, sampler_step=index, noise=noise))
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_rng, actual_rng))
        rows = [json.loads(line) for line in debug.path.read_text().splitlines()][1:]
        self.assertEqual([row["sampler_step"] for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
