import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from scripts.probe_vmem_clip import (
    SCHEMA, attention_profile, compare, difference, digest_file, load_probe, model_fingerprint,
)


class ClipProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, name, offset=0):
        path = self.root / name
        path.mkdir()
        np.savez(path / "samples.npz", preprocessed=np.zeros((3, 1, 3, 2, 2), dtype=np.float32),
                 embeddings=np.full((3, 4), offset, dtype=np.float32))
        report = {"schema": SCHEMA, "status": "complete", "repeats": 3, "attention": "native",
                  "input": "input", "weights": {"sha256": "weights"},
                  "source_sha256": {}, "environment": {},
                  "samples_sha256": digest_file(path / "samples.npz")}
        (path / "report.json").write_text(json.dumps(report))
        return path

    def test_difference_reports_magnitude_without_tolerance(self):
        a = np.array([1., 2.], dtype=np.float32)
        b = np.array([1., 3.], dtype=np.float32)
        row = difference(a, b)
        self.assertFalse(row["equal"])
        self.assertEqual(row["mean_abs"], .5)
        self.assertEqual(row["max_abs"], 1.)
        self.assertAlmostEqual(row["rmse"], np.sqrt(.5))
        self.assertAlmostEqual(row["relative_l2"], 1 / np.sqrt(5))
        self.assertEqual(row["changed_element_fraction"], .5)
        self.assertTrue(difference(a, a)["equal"])

    def test_nonfinite_dtype_and_shape_mismatches_fail(self):
        a = np.zeros((2, 3), dtype=np.float32)
        for b in (a.astype(np.float64), a.T, np.full_like(a, np.nan)):
            with self.assertRaises(ValueError):
                difference(a, b)
        self.assertIsNone(difference(a, a)["relative_l2"])

    def test_compare_arrays_and_input_weight_environment_mismatches(self):
        left, right = self.fixture("left"), self.fixture("right", offset=.125)
        report_path = right / "report.json"
        data = json.loads(report_path.read_text())
        data.update(input="different", weights={"sha256": "changed"}, environment={"version": "changed"})
        report_path.write_text(json.dumps(data))
        report = compare(left, right)
        self.assertFalse(report["input_equal"])
        self.assertFalse(report["weights_equal"])
        self.assertFalse(report["environment_equal"])
        self.assertTrue(report["sources_equal"])
        self.assertTrue(all(row["equal"] for row in report["comparisons"]["preprocessed"]))
        self.assertEqual(report["comparisons"]["embeddings"][0]["max_abs"], .125)

    def test_modified_arrays_are_not_trusted(self):
        path = self.fixture("run")
        with (path / "samples.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_probe(path)

    def test_incomplete_probe_is_not_validated(self):
        path = self.fixture("run")
        report_path = path / "report.json"
        data = json.loads(report_path.read_text())
        data["status"] = "failed"
        report_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "completed"):
            load_probe(path)
        data.update(status="complete", repeats=4)
        report_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Incomplete repetitions"):
            load_probe(path)

    def test_comparison_does_not_rewrite_artifacts(self):
        left, right = self.fixture("left"), self.fixture("right")
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        compare(left, right)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})


class ClipProbeTorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is required")
        cls.torch = torch

    def backend_flags(self):
        torch = self.torch
        return (torch.backends.mha.get_fastpath_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.cudnn_sdp_enabled())

    def conditioner_class(self):
        # Execute the production class, replacing only the heavyweight model factory.
        path = Path(__file__).resolve().parents[1] / "modeling/modules/conditioner.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CLIPConditioner")
        factory = Mock(return_value=(self.torch.nn.Identity(), None, None))
        namespace = {"torch": self.torch, "nn": self.torch.nn,
                     "clip_attention_profile": attention_profile,
                     "open_clip": SimpleNamespace(create_model_and_transforms=factory)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["CLIPConditioner"], factory

    def test_conditioner_defaults_to_native_and_rejects_unknown_before_model_load(self):
        cls, factory = self.conditioner_class()
        with self.assertRaisesRegex(ValueError, "Unknown CLIP attention"):
            cls(attention_profile="invalid")
        factory.assert_not_called()
        self.assertEqual(cls().attention_profile, "native")

    def test_conditioner_scopes_initial_and_batch_encoding_without_precision_or_rng_changes(self):
        torch = self.torch
        cls, _ = self.conditioner_class()
        for profile in ("native", "math"):
            for batch, autocast in ((1, False), (4, True)):
                with self.subTest(profile=profile, batch=batch):
                    encoder = cls(attention_profile=profile)
                    before = self.backend_flags()
                    expected = before if profile == "native" else (False, False, False, True, False)
                    calls = []

                    def checked(value):
                        calls.append(value)
                        self.assertEqual(self.backend_flags(), expected)
                        self.assertEqual(torch.is_autocast_enabled("cpu"), autocast)
                        self.assertEqual(torch.is_grad_enabled(), not autocast)
                        return value

                    encoder.preprocess = checked
                    encoder.module.encode_image = checked
                    source = torch.arange(batch * 12, dtype=torch.float32).reshape(batch, 3, 2, 2)
                    rng = torch.get_rng_state().clone()
                    precision = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
                    with torch.set_grad_enabled(not autocast), torch.autocast("cpu", enabled=autocast):
                        self.assertIs(encoder(source), source)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(self.backend_flags(), before)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                    self.assertEqual(precision, (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()))

    def test_conditioner_restores_dispatch_after_preprocess_or_encode_failure(self):
        cls, _ = self.conditioner_class()
        for stage in ("preprocess", "encode_image"):
            with self.subTest(stage=stage):
                encoder = cls(attention_profile="math")
                encoder.preprocess = lambda value: value
                encoder.module.encode_image = lambda value: value
                target = encoder if stage == "preprocess" else encoder.module
                setattr(target, stage, Mock(side_effect=RuntimeError("encoding failed")))
                before = self.backend_flags()
                with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                    encoder(self.torch.ones(1, 3, 2, 2))
                self.assertEqual(self.backend_flags(), before)

    def test_math_attention_restores_backend_flags_on_failure(self):
        torch = self.torch
        before = (torch.backends.mha.get_fastpath_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                  torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with attention_profile("math", torch):
                self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                raise RuntimeError("test failure")
        after = (torch.backends.mha.get_fastpath_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                 torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
        self.assertEqual(before, after)

    def test_native_profile_leaves_flags_alone(self):
        before = self.torch.backends.mha.get_fastpath_enabled()
        with attention_profile("native", self.torch):
            self.assertEqual(self.torch.backends.mha.get_fastpath_enabled(), before)

    def test_model_hash_includes_nonpersistent_preprocessing_buffers(self):
        from generation_debug import GenerationDebug
        torch = self.torch
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
        model.register_buffer("mean", torch.zeros(3), persistent=False)
        with tempfile.TemporaryDirectory() as tmp:
            debug = GenerationDebug(Path(tmp) / "probe.jsonl", seed=1, mode="observe", device="cpu", torch_module=torch)
            first = model_fingerprint(model, debug)
            self.assertEqual(first, model_fingerprint(model, debug))
            self.assertIn("buffer:mean", first["tensors"])
            model.mean[0] = 1
            self.assertNotEqual(first["sha256"], model_fingerprint(model, debug)["sha256"])


if __name__ == "__main__":
    unittest.main()
