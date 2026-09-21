import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_script("build_vmem_transfer_manifest")
RUNNER = load_script("run_vmem_demo_actions")
LAUNCHER = load_script("run_vmem_demo_manifest")


class TransferGenerationTest(unittest.TestCase):
    def test_frozen_manifest_matches_builder_and_pairs(self):
        path = ROOT / "manifests" / "vmem_transfer_v1.jsonl"
        rows = LAUNCHER._load_manifest(path)
        actual = [{k: v for k, v in row.items() if k != "_manifest_line"} for row in rows]
        self.assertEqual(actual, BUILDER.build_rows())
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({row["run_id"] for row in rows}), 30)
        for i in range(0, len(rows), 2):
            baseline, geocov = rows[i:i + 2]
            ignored = {"run_id", "memory_policy", "memory_budget", "_manifest_line"}
            self.assertEqual(
                {k: v for k, v in baseline.items() if k not in ignored},
                {k: v for k, v in geocov.items() if k not in ignored},
            )
            self.assertEqual(baseline["memory_policy"], "unbounded")
            self.assertNotIn("memory_budget", baseline)
            self.assertEqual(geocov["memory_policy"], "slam_covisibility")
            self.assertEqual(geocov["memory_budget"], 32)
            self.assertTrue((ROOT / baseline["image"]).is_file())
            for row in (baseline, geocov):
                command = LAUNCHER._command_for_row(
                    row, output_root=Path("outputs/transfer"),
                    config=Path("configs/inference/inference.yaml"), device="cuda", dry_run=True,
                )
                self.assertIn("--dry-run", command)
                self.assertNotIn("--_case_id", command)

    def test_all_paths_close_and_fit_sixty_seconds(self):
        for name, step_size in BUILDER.TRAJECTORIES:
            args = SimpleNamespace(trajectory=name, num_actions=195, seed=501)
            actions = RUNNER._expand_trajectory_actions(args)
            self.assertEqual(len(actions), 195)
            self.assertEqual(RUNNER._num_actions_from_duration(60, fps=13, frames_per_action=4), 195)
            period = 40 if name == "out_and_back" else 36
            rotation = sum({"left5": 5, "right5": -5, "left10": 10, "right10": -10}.get(a, 0)
                           for a in actions[:period])
            translation = sum({"forward": -step_size, "backward": step_size}.get(a, 0)
                              for a in actions[:period])
            self.assertAlmostEqual(rotation, 0)
            self.assertAlmostEqual(translation, 0)
            if name == "out_and_back":
                self.assertAlmostEqual(20 * step_size, 0.4)

    def test_provenance_identifies_changed_input_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            image, config = Path(tmp) / "image", Path(tmp) / "config"
            image.write_bytes(b"first")
            config.write_text("model: test\n")
            first = RUNNER._generation_provenance(image, config)
            image.write_bytes(b"second")
            second = RUNNER._generation_provenance(image, config)
            self.assertNotEqual(first["image_sha256"], second["image_sha256"])
            self.assertEqual(first["config_sha256"], second["config_sha256"])
            self.assertEqual(first["source_sha256"], second["source_sha256"])
            self.assertIn("modeling/pipeline.py", first["source_sha256"])
            self.assertIn("clip_attention.py", first["source_sha256"])

    def test_navigator_initialization_preserves_other_run_outputs(self):
        # Execute the real initialization method without importing CUDA dependencies.
        module = ast.parse((ROOT / "navigation.py").read_text())
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Navigator")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "initialize")
        namespace = {"os": os, "np": np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "navigation.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "visualization"
            shared.mkdir()
            sentinel = shared / "other_job.txt"
            sentinel.write_text("keep")
            own_dir = root / "run" / "visualization"
            pipeline = SimpleNamespace(visualize_dir=str(own_dir), initialize=lambda *args: "initial_frame")
            navigator = SimpleNamespace(pipeline=pipeline, pose_history=[])
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                result = namespace["initialize"](navigator, None, np.eye(4), np.eye(3))
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(result, "initial_frame")
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertTrue(own_dir.is_dir())

    def test_headless_navigator_does_not_keep_initial_image_alias(self):
        module = ast.parse((ROOT / "navigation.py").read_text())
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Navigator")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "initialize")
        namespace = {"os": os, "np": np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "navigation.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = SimpleNamespace(visualize_dir=tmp, initialize=lambda *args: "image")
            navigator = SimpleNamespace(pipeline=pipeline, pose_history=[], retain_frame_history=False)
            self.assertEqual(namespace["initialize"](navigator, None, np.eye(4), np.eye(3)), "image")
            self.assertEqual(navigator.frames, [])


if __name__ == "__main__":
    unittest.main()
