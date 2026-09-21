import json
from pathlib import Path
import tempfile
import unittest

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
