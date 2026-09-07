import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_vmem_revisits", ROOT / "scripts" / "evaluate_vmem_revisits.py"
)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def yaw_pose(degrees, translation=(0, 0, 0)):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    pose[:3, 3] = translation
    return pose.tolist()


def records_for_poses(poses):
    return [
        {"action_index": i, "num_generated_frames": 4,
         "total_frames_after": 1 + 4 * (i + 1), "current_pose": pose}
        for i, pose in enumerate(poses)
    ]


class RevisitTest(unittest.TestCase):
    def test_counts_all_returns_once_per_excursion(self):
        records = records_for_poses([yaw_pose(a) for a in (0, 5, 0, 0, -5, 0)])
        self.assertEqual([r["frame_index"] for r in EVAL.find_returns(records)], [12, 24])

    def test_nearest_non_return_is_rejected(self):
        records = records_for_poses([yaw_pose(a) for a in (5, 10, 1)])
        self.assertEqual(EVAL.find_returns(records), [])

    def test_identity_rotation_with_translation_is_not_a_return(self):
        records = records_for_poses([yaw_pose(20), yaw_pose(0, (0.03, 0, 0))])
        self.assertEqual(EVAL.find_returns(records), [])

    def test_pan45_sixty_seconds_has_ten_returns(self):
        steps = ([5] * 9 + [-5] * 18 + [5] * 9) * 6
        yaws = np.cumsum(steps[:195])
        returns = EVAL.find_returns(records_for_poses([yaw_pose(a) for a in yaws]))
        self.assertEqual([r["frame_index"] for r in returns], list(range(72, 721, 72)))

    def test_invalid_pose_rejected(self):
        records = records_for_poses([np.full((4, 4), np.nan).tolist()])
        with self.assertRaises(ValueError):
            EVAL.find_returns(records)

    def test_pixel_mse_normalization_and_no_resize(self):
        anchor = np.zeros((576, 576, 3), dtype=np.uint8)
        frame = anchor.copy()
        frame[0, 0] = 255
        self.assertAlmostEqual(EVAL.image_metrics(anchor, frame)["rgb_mse"], 1 / (576 * 576))
        self.assertEqual(EVAL.image_metrics(anchor, anchor)["rgb_mse"], 0)

    def test_retrieval_join_uses_target_frame_id(self):
        trace = [{"target_frame_indices": [69, 70, 71, 72],
                  "allowed_memory_indices": [0, 2, 3], "selected_context_indices": [2, 3]}]
        found = EVAL.retrieval_for_frame(trace, 72)
        self.assertTrue(found["anchor_retained"])
        self.assertFalse(found["anchor_selected"])
        self.assertFalse(EVAL.retrieval_for_frame(trace, 144)["retrieval_logged"])

    def test_run_match_rejects_wrong_seed_duration_and_policy(self):
        row = {"run_id": "oxford", "image": "test_samples/oxford.jpg",
               "trajectory": "pan_45", "duration_seconds": 60, "seed": 501}
        metadata = {**row, "num_actions": 195, "actual_frames": 781}
        self.assertTrue(EVAL.metadata_matches(metadata, row))
        for change in ({"seed": 42}, {"num_actions": 33}, {"actual_frames": 77},
                       {"memory_policy": "slam_covisibility"}):
            self.assertFalse(EVAL.metadata_matches({**metadata, **change}, row))
        legacy = dict(metadata)
        del legacy["seed"]
        self.assertTrue(EVAL.metadata_matches(legacy, row))

    def test_latest_matching_run_skips_prefix_collision(self):
        row = {"run_id": "oxford", "image": "test_samples/oxford.jpg", "num_actions": 3}
        metadata = {**row, "actual_frames": 13, "pattern": "forward"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correct = root / "oxford_unbounded"
            correct.mkdir()
            (correct / "metadata.json").write_text(json.dumps(metadata))
            wrong = root / "oxford_slam32"
            wrong.mkdir()
            (wrong / "metadata.json").write_text(json.dumps({**metadata, "memory_policy": "slam_covisibility"}))
            self.assertEqual(EVAL.find_run(root, row)[0], correct)

    def test_cpu_png_scoring_writes_all_returns_and_last_error(self):
        from PIL import Image

        row = {"run_id": "fixture", "image": "fixture.jpg", "trajectory": "pan_45",
               "num_actions": 4, "seed": 42}
        metadata = {**row, "actual_frames": 17, "fps": 13}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "fixture_unbounded"
            frames = run / "generated_frames"
            frames.mkdir(parents=True)
            (run / "metadata.json").write_text(json.dumps(metadata))
            actions = records_for_poses([yaw_pose(a) for a in (5, 0, -5, 0)])
            (run / "actions.json").write_text(json.dumps(actions))
            trace = [
                {"target_frame_indices": [8], "allowed_memory_indices": [0, 2], "selected_context_indices": [0]},
                {"target_frame_indices": [16], "allowed_memory_indices": [0, 3], "selected_context_indices": [3]},
            ]
            (run / "retrieval_trace.json").write_text(json.dumps(trace))
            for index, value in ((0, 0), (8, 0), (16, 255)):
                Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(frames / f"{index:04d}.png")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "evaluate_vmem_revisits.py"),
                 str(manifest), "--output-root", str(root), "--csv-out", str(root / "scores.csv")],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "scores.csv").open() as handle:
                summary = list(csv.DictReader(handle))[0]
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(float(summary["mean_rgb_mse"]), 0.5)
            self.assertEqual(float(summary["last_rgb_mse"]), 1)
            self.assertEqual(float(summary["mean_anchor_selected"]), 0.5)
            self.assertEqual(summary["frame_source"], "png")
            with (root / "scores.returns.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_pilot_matched_inputs_and_equal_budgets(self):
        rows = EVAL.load_rows(ROOT / "manifests" / "vmem_geocov_pilot.jsonl")
        self.assertEqual(len(rows), 9)
        for start in range(0, len(rows), 3):
            group = rows[start:start + 3]
            self.assertEqual([r["memory_policy"] for r in group],
                             ["unbounded", "slam_covisibility", "rarity_irreplaceability"])
            inputs = [{k: v for k, v in r.items() if k not in
                       {"run_id", "memory_policy", "memory_budget"}} for r in group]
            self.assertEqual(inputs[0], inputs[1])
            self.assertEqual(inputs[0], inputs[2])
            self.assertEqual(group[1]["memory_budget"], group[2]["memory_budget"])
            self.assertTrue((ROOT / group[0]["image"]).exists())


if __name__ == "__main__":
    unittest.main()
