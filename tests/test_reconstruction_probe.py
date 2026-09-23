import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from scripts.probe_vmem_reconstruction import (
    SCHEMA, STAGES, array_difference, attention_state, compare, compare_stage, digest_file,
    first_action_poses, load_reference, load_report, main, observe_reconstruction, run,
    snapshot, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
ARRAY_ONLY = SimpleNamespace(Tensor=type("UnusedTensor", (), {}))


def attention_record(profile):
    before = {"mha_fastpath": True, "flash_sdp": True, "memory_efficient_sdp": True,
              "math_sdp": True, "cudnn_sdp": True}
    active = before.copy() if profile == "native" else {
        "mha_fastpath": False, "flash_sdp": False, "memory_efficient_sdp": False,
        "math_sdp": True, "cudnn_sdp": False}
    return {"profile": profile, "before": before, "active": active, "after": before.copy()}


class ReconstructionProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def reference(self):
        path = self.root / "reference"
        path.mkdir()
        write_json(path / "run_status.json", {"status": "complete"})
        write_json(path / "run_spec.json", {"arguments": {"num_actions": 1, "frames_per_action": 4},
                                            "actions": ["left5"]})
        write_json(path / "actions.json", [{"action_index": 0, "action": "left5", "total_frames_after": 5}])
        (path / "resource_trace.jsonl").write_text(json.dumps({"event": "step", "step": 0,
                                                              "reconstruction_input_indices": list(range(5))}) + "\n")
        (path / "generation_config.yaml").write_text("surfel: {}\n")
        write_json(path / "generation_environment.json", {})
        (path / "generated_frames").mkdir()
        hashes = []
        for index in range(5):
            image_path = path / "generated_frames" / f"{index:04d}.png"
            with Image.new("RGB", (4, 4), (index, 0, 0)) as image:
                image.save(image_path)
            hashes.append(digest_file(image_path))
        write_json(path / "frame_manifest.json", {"schema": "vmem_output_frames_v1", "sha256": hashes})
        return path

    def probe(self, name, *, changed_stage=None, attention=None):
        path = self.root / name
        path.mkdir()
        repetitions = []
        for index in range(2):
            target = path / f"repeat_{index:02d}"
            target.mkdir()
            stages = {}
            for stage in STAGES:
                value = np.ones((2, 3), dtype=np.float32)
                if stage == changed_stage:
                    value[0, 0] += .25
                record, arrays = snapshot({"value": value, "mask": np.array([np.nan], dtype=np.float32)}, ARRAY_ONLY)
                artifact = target / f"{stage}.npz"
                np.savez_compressed(artifact, **arrays)
                stages[stage] = {**record, "artifact_sha256": digest_file(artifact)}
            repetitions.append({"index": index, "stages": stages,
                                "rng": {"start": "same", "before_alignment": "same", "end": "same"}})
        report = {"schema": SCHEMA, "status": "complete", "settings": {"repeats": 2},
                  "reference_inputs": {"hash": "same"}, "weights": {"sha256": "same"},
                  "source_sha256": {"file": "same"}, "environment": {"test": True},
                  "reconstruction_environment": {"test": True},
                  "weights_unchanged": True, "repetitions": repetitions}
        if attention is not None:
            report["settings"]["prediction_attention"] = attention
            for record in repetitions:
                record["prediction_attention"] = attention_record(attention)
        write_json(path / "report.json", report)
        return path

    def edit(self, path, function):
        record = json.loads(path.read_text())
        function(record)
        write_json(path, record)

    def test_reference_checks_exact_rgb_prefix_and_leaves_files_untouched(self):
        path = self.reference()
        before = {str(file): file.read_bytes() for file in path.rglob("*") if file.is_file()}
        _, _, images, provenance = load_reference(path)
        try:
            self.assertEqual(len(images), 5)
            self.assertEqual(images[-1].getpixel((0, 0)), (4, 0, 0))
            self.assertEqual(provenance["frame_indices"], list(range(5)))
        finally:
            for image in images:
                image.close()
        self.assertEqual(before, {str(file): file.read_bytes() for file in path.rglob("*") if file.is_file()})

    def test_changed_saved_png_is_rejected_before_runtime_load(self):
        path = self.reference()
        with (path / "generated_frames/0000.png").open("ab") as handle:
            handle.write(b"changed")
        from scripts import run_vmem_demo_actions as runner
        with patch.object(runner, "_load_runtime_dependencies") as runtime, self.assertRaisesRegex(ValueError, "hash mismatch"):
            run(SimpleNamespace(reference_run=path, output=self.root / "probe"))
        runtime.assert_not_called()
        self.assertFalse((self.root / "probe").exists())

    def test_non_rgb_images_and_partial_reference_are_rejected(self):
        path = self.reference()
        image_path = path / "generated_frames/0000.png"
        with Image.new("L", (4, 4)) as image:
            image.save(image_path)
        self.edit(path / "frame_manifest.json", lambda row: row["sha256"].__setitem__(0, digest_file(image_path)))
        with self.assertRaisesRegex(ValueError, "RGB PNG"):
            load_reference(path)
        write_json(path / "run_status.json", {"status": "running"})
        with self.assertRaisesRegex(ValueError, "complete"):
            load_reference(path)

    def test_different_first_reconstruction_inputs_are_rejected(self):
        path = self.reference()
        (path / "resource_trace.jsonl").write_text(json.dumps({"event": "step", "step": 0,
                                                              "reconstruction_input_indices": [0, 1, 3, 4]}) + "\n")
        with self.assertRaisesRegex(ValueError, "frames 0-4"):
            load_reference(path)

    def test_output_cannot_be_inside_reference(self):
        path = self.reference()
        with self.assertRaisesRegex(ValueError, "outside"):
            run(SimpleNamespace(reference_run=path, output=path / "probe"))
        self.assertFalse((path / "probe").exists())

    def test_runtime_failure_is_saved_and_not_marked_complete(self):
        path = self.reference()
        output = self.root / "failed"
        with patch("scripts.probe_vmem_reconstruction._run_loaded", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                run(SimpleNamespace(reference_run=path, output=output))
        self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "failed")
        with self.assertRaisesRegex(ValueError, "Not a completed"):
            load_report(output)
        before = (output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run(SimpleNamespace(reference_run=path, output=output))
        self.assertEqual(before, (output / "report.json").read_bytes())

    def test_nan_placeholders_and_real_numeric_error_are_distinguished(self):
        a = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        self.assertTrue(array_difference(a, a)["values_equal_nan_aware"])
        b = a.copy()
        b[-1] = 1.5
        result = array_difference(a, b)
        self.assertFalse(result["values_equal_nan_aware"])
        self.assertEqual(result["finite_elements_compared"], 1)
        self.assertEqual(result["max_abs_finite"], .5)
        self.assertEqual(result["changed_element_fraction"], .25)
        self.assertFalse(array_difference(a, b.astype(np.float64))["comparable"])

    def test_first_differing_stage_and_fixed_controls(self):
        left, right = self.probe("left"), self.probe("right", changed_stage="predictions")
        report = compare(left, right)
        self.assertTrue(report["matched_control"])
        self.assertTrue(all(row["first_different_stage"] == "predictions" for row in report["comparisons"]))
        first = report["comparisons"][0]["stages"]["predictions"]["first_difference"]
        self.assertEqual(first["errors"]["max_abs_finite"], .25)
        self.edit(right / "report.json", lambda row: row.update(weights_unchanged=False))
        self.assertFalse(compare(left, right)["matched_control"])

    def test_snapshot_preserves_tensor_values_dtype_and_rng(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch is required")
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        before, rng = value.clone(), torch.get_rng_state().clone()
        record, arrays = snapshot({"value": value, "label": None}, torch)
        self.assertEqual(record["arrays"]["array_0000"]["dtype"], "float32")
        self.assertEqual(record["arrays"]["array_0000"]["shape"], [4, 3])
        self.assertTrue(torch.equal(value, before))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        np.testing.assert_array_equal(arrays["array_0000"], value.numpy())
        with self.assertRaisesRegex(ValueError, "dtype"):
            snapshot(np.array([object()], dtype=object), torch)


    def test_same_process_repeats_use_different_artifact_paths(self):
        path = self.probe("same")
        report = load_report(path)
        a, b = report["repetitions"]
        self.assertTrue(compare_stage(path, path, a["stages"]["aligned"], b["stages"]["aligned"],
                                      0, "aligned", right_repeat=1)["equal"])
        with (path / "repeat_01/aligned.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed stage artifact"):
            compare_stage(path, path, a["stages"]["aligned"], b["stages"]["aligned"],
                          0, "aligned", right_repeat=1)

    def test_updated_file_digest_cannot_hide_stale_array_fingerprints(self):
        left, right = self.probe("left"), self.probe("right")
        artifact = right / "repeat_00/aligned.npz"
        with np.load(artifact, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["array_0001"][0, 0] += 1
        np.savez_compressed(artifact, **arrays)
        self.edit(right / "report.json", lambda row: row["repetitions"][0]["stages"]["aligned"].update(
            artifact_sha256=digest_file(artifact)))
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            compare(left, right)

    def test_partial_stages_are_not_equal_and_comparison_is_read_only(self):
        left, right = self.probe("left"), self.probe("right")
        before = {str(file): file.read_bytes() for file in self.root.rglob("*") if file.is_file()}
        self.assertTrue(compare(left, right)["matched_control"])
        self.assertEqual(before, {str(file): file.read_bytes() for file in self.root.rglob("*") if file.is_file()})
        self.edit(right / "report.json", lambda row: row["repetitions"][0]["stages"].pop("predictions"))
        with self.assertRaisesRegex(ValueError, "Incomplete reconstruction stages"):
            compare(left, right)

    def test_environment_and_rng_mismatches_are_not_a_matched_control(self):
        left, right = self.probe("left"), self.probe("right")
        self.edit(right / "report.json", lambda row: row["repetitions"][0]["rng"].update(start="changed"))
        self.assertFalse(compare(left, right)["matched_control"])
        self.edit(right / "report.json", lambda row: row["environment"].update(test=False))
        self.assertFalse(compare(left, right)["environment_equal"])

    def test_missing_rng_or_provenance_is_not_a_matched_control(self):
        left, right = self.probe("left"), self.probe("right")
        self.edit(right / "report.json", lambda row: row["repetitions"][0]["rng"].pop("before_alignment"))
        with self.assertRaisesRegex(ValueError, "RNG records"):
            compare(left, right)
        self.edit(right / "report.json", lambda row: row.update(weights={}))
        with self.assertRaisesRegex(ValueError, "provenance"):
            compare(left, right)

    def test_prediction_attention_mismatch_is_not_a_matched_control(self):
        left = self.probe("left", attention="math")
        right = self.probe("right", attention="math")
        self.assertTrue(compare(left, right)["matched_control"])
        native = self.probe("native", attention="native")
        report = compare(native, right)
        self.assertFalse(report["matched_control"])
        self.assertFalse(report["settings_equal"])
        self.assertEqual(report["prediction_attention"], ["native", "math"])
        self.assertTrue(all(not row["prediction_attention_equal"] for row in report["comparisons"]))

    def test_missing_or_unrestored_attention_record_is_rejected(self):
        for case in ("missing", "wrong_active", "unrestored", "not_boolean"):
            with self.subTest(case=case):
                path = self.probe(case, attention="math")

                def damage(report):
                    trace = report["repetitions"][0]["prediction_attention"]
                    if case == "missing":
                        trace.pop("active")
                    elif case == "wrong_active":
                        trace["active"]["flash_sdp"] = True
                    elif case == "unrestored":
                        trace["after"]["mha_fastpath"] = False
                    else:
                        trace["active"]["math_sdp"] = 1

                self.edit(path / "report.json", damage)
                with self.assertRaisesRegex(ValueError, "attention"):
                    load_report(path)

    def test_unknown_attention_is_rejected_before_loading_inputs(self):
        with patch("scripts.probe_vmem_reconstruction.load_reference") as load:
            with self.assertRaisesRegex(ValueError, "Unknown prediction attention"):
                run(SimpleNamespace(attention="invalid"))
            load.assert_not_called()

    def test_cli_attention_is_opt_in_and_passed_to_runner(self):
        for profile in (None, "math"):
            argv = ["probe", "run", "--reference-run", "reference", "--output", "probe"]
            if profile is not None:
                argv.extend(["--attention", profile])
            with patch("sys.argv", argv), patch("scripts.probe_vmem_reconstruction.run") as launch:
                main()
            self.assertEqual(launch.call_args.args[0].attention, profile or "native")

    def test_observation_preserves_objects_and_restores_functions_on_failure(self):
        views = [{"img": np.ones((1, 3, 2, 2))}]
        predictions = [{"conf": np.ones((2, 2))}]
        result = {key: [np.ones((2, 2))] for key in ("point_clouds", "depths", "confidences", "camera_info")}
        prepare = Mock(return_value=views)
        infer = Mock(return_value=({"pred": predictions}, None))
        align = Mock(return_value=result)
        reconstruction = SimpleNamespace(prepare_input_from_pil=prepare, prepare_output=align)
        inference_module = SimpleNamespace(inference=infer)
        inputs, poses = ["image"], np.eye(4)

        def actual_call(images, model, **kwargs):
            self.assertIs(images, inputs)
            self.assertIs(kwargs["poses"], poses)
            self.assertIsNone(kwargs["depths"])
            self.assertEqual(kwargs["size"], 512)
            self.assertEqual(kwargs["niter"], 400)
            self.assertIs(reconstruction.prepare_input_from_pil(), views)
            self.assertIs(inference_module.inference()[0]["pred"], predictions)
            return reconstruction.prepare_output()

        reconstruction.run_inference_from_pil = actual_call
        debug = SimpleNamespace(rng_state=lambda: "unchanged", fingerprint=lambda value: value)
        captured = []
        rng = observe_reconstruction(reconstruction, inference_module, object(), inputs, poses=poses,
                                     lr=.01, niter=400, device="cpu", capture=lambda *args: captured.append(args), debug=debug)
        self.assertEqual([name for name, _ in captured], list(STAGES))
        self.assertIs(captured[0][1], views)
        self.assertIs(captured[1][1], predictions)
        self.assertEqual(set(rng), {"start", "before_alignment", "end"})
        with self.assertRaisesRegex(RuntimeError, "capture failure"):
            observe_reconstruction(reconstruction, inference_module, object(), inputs, poses=poses,
                                   lr=.01, niter=400, device="cpu", debug=debug,
                                   capture=Mock(side_effect=RuntimeError("capture failure")))
        self.assertIs(reconstruction.prepare_input_from_pil, prepare)
        self.assertIs(reconstruction.prepare_output, align)
        self.assertIs(inference_module.inference, infer)

    def test_real_navigator_pose_replay_and_float32_convention(self):
        import scipy.spatial.transform as spt
        from scripts.run_vmem_demo_actions import _apply_action
        tree = ast.parse((ROOT / "navigation.py").read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Navigator")
        # Keep real action/interpolation methods without importing model modules.
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), original], type_ignores=[])
        namespace = {"np": np, "spt": spt}
        exec(compile(ast.fix_missing_locations(module), "navigation.py", "exec"), namespace)
        pipeline_tree = ast.parse((ROOT / "modeling/pipeline.py").read_text())
        pipeline_cls = next(node for node in pipeline_tree.body if isinstance(node, ast.ClassDef) and node.name == "VMemPipeline")
        method = next(node for node in pipeline_cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_transformed_c2ws")
        from copy import deepcopy
        pose_namespace = {"np": np, "deepcopy": deepcopy}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "pipeline.py", "exec"), pose_namespace)
        runner = SimpleNamespace(Navigator=namespace["Navigator"], _apply_action=_apply_action,
                                 get_default_intrinsics=lambda: [np.eye(3)],
                                 VMemPipeline=SimpleNamespace(get_transformed_c2ws=pose_namespace["get_transformed_c2ws"]))
        spec = {"arguments": {"step_size": .1}}
        expected = np.eye(4)
        expected[:3, :3] = spt.Rotation.from_euler("y", 5, degrees=True).as_matrix()
        poses = first_action_poses(runner, spec, {"action": "left5", "current_pose": expected.tolist()})
        self.assertEqual(poses.shape, (5, 4, 4))
        self.assertEqual(poses.dtype, np.float32)
        np.testing.assert_array_equal(poses[0], np.diag([1, -1, -1, 1]))
        expected[:, [1, 2]] *= -1
        np.testing.assert_allclose(poses[-1], expected, rtol=0, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            first_action_poses(runner, spec, {"action": "left5", "current_pose": np.eye(4).tolist()})


class ReconstructionAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is required")
        cls.torch = torch

    def exercise(self, profile, fail=None):
        torch = self.torch
        before, rng = attention_state(torch), torch.get_rng_state().clone()
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
        active = before if profile == "native" else attention_record("math")["active"]
        visited, trace = [], {}
        views, predictions = [object()], [object()]
        result = {key: object() for key in ("point_clouds", "depths", "confidences", "camera_info")}

        def check(stage, expected):
            visited.append(stage)
            self.assertEqual(attention_state(torch), expected)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            if stage == fail:
                raise RuntimeError("injected failure")

        def prepare():
            check("preprocessing", before)
            return views

        def infer():
            check("inference", active)
            return {"pred": predictions}, None

        def align():
            check("alignment", before)
            return result

        def capture(stage, value):
            check("capture_" + stage, before)
            if stage == "predictions":
                self.assertIs(value, predictions)

        reconstruction = SimpleNamespace(prepare_input_from_pil=prepare, prepare_output=align)
        inference_module = SimpleNamespace(inference=infer)

        def actual_call(images, model, **kwargs):
            self.assertIs(reconstruction.prepare_input_from_pil(), views)
            self.assertIs(inference_module.inference()[0]["pred"], predictions)
            return reconstruction.prepare_output()

        reconstruction.run_inference_from_pil = actual_call
        debug = SimpleNamespace(rng_state=lambda: "same", fingerprint=lambda value: value)
        try:
            with torch.no_grad(), torch.autocast("cpu"):
                observe_reconstruction(reconstruction, inference_module, object(), ["image"],
                                       poses=np.eye(4), lr=.01, niter=400, device="cpu", capture=capture,
                                       debug=debug, attention=profile, torch_module=torch, attention_trace=trace)
        finally:
            self.assertEqual(attention_state(torch), before)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(precision, (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()))
            self.assertEqual(trace, {"profile": profile, "before": before, "active": active, "after": before})
            self.assertIs(reconstruction.prepare_input_from_pil, prepare)
            self.assertIs(reconstruction.prepare_output, align)
            self.assertIs(inference_module.inference, infer)
        self.assertEqual(visited, ["preprocessing", "capture_preprocessed", "inference", "capture_predictions",
                                   "alignment", "capture_aligned"])

    def test_profile_is_scoped_to_inference_without_rng_or_precision_changes(self):
        for profile in ("native", "math"):
            with self.subTest(profile=profile):
                self.exercise(profile)

    def test_profile_and_wrappers_restore_after_inference_capture_or_alignment_failure(self):
        for fail in ("inference", "capture_predictions", "alignment"):
            with self.subTest(fail=fail), self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.exercise("math", fail=fail)


if __name__ == "__main__":
    unittest.main()
