import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "modeling" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


def make_line_c2ws(positions):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    c2ws[:, 0, 3] = np.array(positions, dtype=np.float64)
    return c2ws


class KCenterCoresetTest(unittest.TestCase):
    def test_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.compute_kcenter_coreset_scores(
                memory_frame_indices=[0, 1],
                c2ws=make_line_c2ws([0.0, 1.0]),
                budget=None,
            )

    def test_prefers_spread_over_near_duplicates(self):
        # a1, a2 near-identical positions; b far away. Budget 2 should keep
        # one of {a1, a2} plus b, minimizing the maximum distance from any
        # archive point to its nearest retained center -- not both near
        # duplicates, which would leave b's whole neighborhood uncovered.
        c2ws = make_line_c2ws([0.0, 0.1, 20.0])

        scores, details = MEMORY_POLICIES.compute_kcenter_coreset_scores(
            memory_frame_indices=[0, 1, 2],
            c2ws=c2ws,
            budget=2,
            pose_weight=1.0,
            visual_weight=0.0,
            return_details=True,
        )

        selected = {frame_idx for frame_idx, row in details.items() if row["kcenter_selected"]}
        self.assertIn(2, selected)
        self.assertEqual(len(selected), 2)
        self.assertNotEqual(selected, {0, 1})

    def test_forced_frames_are_never_evicted(self):
        c2ws = make_line_c2ws([0.0, 0.05, 20.0])

        _, details = MEMORY_POLICIES.compute_kcenter_coreset_scores(
            memory_frame_indices=[0, 1, 2],
            c2ws=c2ws,
            budget=1,
            forced_keep_frames={1},
            pose_weight=1.0,
            visual_weight=0.0,
            return_details=True,
        )
        self.assertTrue(details[1]["kcenter_selected"])
        self.assertTrue(details[1]["kcenter_forced_keep"])
        self.assertEqual(details[1]["score"], float("inf"))

    def test_under_budget_keeps_everything(self):
        c2ws = make_line_c2ws([0.0, 1.0])
        scores, details = MEMORY_POLICIES.compute_kcenter_coreset_scores(
            memory_frame_indices=[0, 1],
            c2ws=c2ws,
            budget=5,
            pose_weight=1.0,
            visual_weight=0.0,
            return_details=True,
        )
        self.assertTrue(all(row["kcenter_selected"] for row in details.values()))


if __name__ == "__main__":
    unittest.main()
