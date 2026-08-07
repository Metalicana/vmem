import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "modeling" / "memory_policies.py"
SPEC = importlib.util.spec_from_file_location("memory_policies", MODULE_PATH)
MEMORY_POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY_POLICIES)


def dino_features(values):
    return {frame_idx: np.array(value, dtype=np.float32) for frame_idx, value in values.items()}


class MarginalCoverageEvictionTest(unittest.TestCase):
    """Sanity checks for VMem's MCE kernel/reverse-deletion core.

    K_geo is precomputed here (not rendered from a real surfel index) since
    ``compute_marginal_coverage_eviction_scores`` takes it as a plain matrix
    -- the backbone-agnostic split agreed for VMem's adapter. This exercises
    exactly the same Algorithm 1 logic as MemCam's ``test_mce.py``; only the
    K_geo source differs between backbones.
    """

    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            MEMORY_POLICIES.FrameMemoryBuffer(policy="mce")

    def test_paper_table1_duplicate_view_counterexample(self):
        # a1, a2: near-identical room-A views (high mutual K_geo). b: distinct
        # room-B view (K_geo ~0 to both). Budget 2 should keep one room-A
        # view plus b, not both near-duplicate room-A views.
        memory_frame_indices = [0, 1, 2]
        hist_query_frame_indices = [0, 1, 2]  # one medoid per candidate here
        # rows = queries (0,1,2), cols = candidates (0,1,2)
        hist_geo_matrix = np.array(
            [
                [1.0, 0.95, 0.0],
                [0.95, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dino = dino_features(
            {
                0: [1.0, 0.0],
                1: [0.98, 0.02],
                2: [0.0, 1.0],
            }
        )

        scores, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=memory_frame_indices,
            budget=2,
            hist_query_frame_indices=hist_query_frame_indices,
            hist_geo_matrix=hist_geo_matrix,
            dino_features=dino,
            return_details=True,
        )

        selected = {frame_idx for frame_idx, row in details.items() if row["mce_selected"]}
        self.assertEqual(selected, {0, 2})
        self.assertLess(scores[1], scores[0])
        self.assertLess(scores[1], scores[2])

        memory = MEMORY_POLICIES.FrameMemoryBuffer(policy="mce", budget=2)
        evicted = memory.update([0, 1, 2], eviction_scores=scores)
        self.assertEqual(memory.candidates(), [0, 2])
        self.assertEqual(evicted, [1])

    def test_kernel_is_additive_not_multiplicative(self):
        # Two candidates: identical geometry (K_geo=1 both ways), orthogonal
        # appearance. Under the paper's additive kernel
        # alpha*K_geo + (1-alpha)*K_vis, K_geo=1 keeps K(q,m) well above zero
        # even though K_vis=0 -- a multiplicative combination would collapse
        # it to ~0.
        hist_geo_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        dino = dino_features({0: [1.0, 0.0], 1: [0.0, 1.0]})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            budget=2,
            hist_query_frame_indices=[0, 1],
            hist_geo_matrix=hist_geo_matrix,
            dino_features=dino,
            alpha=0.65,
            return_details=True,
        )
        self.assertGreater(details[0]["mce_coverage_value"], 0.7)

    def test_forced_frames_are_never_evicted(self):
        hist_geo_matrix = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dino = dino_features({0: [1.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1, 2],
            budget=1,
            hist_query_frame_indices=[0, 1, 2],
            hist_geo_matrix=hist_geo_matrix,
            forced_keep_frames={1},
            dino_features=dino,
            return_details=True,
        )
        self.assertTrue(details[1]["mce_selected"])
        self.assertTrue(details[1]["mce_forced_keep"])
        self.assertEqual(details[1]["score"], float("inf"))

    def test_future_query_biases_selection_toward_reachable_view(self):
        # Two candidates, no historical reason to prefer either (K_geo/K_vis
        # both neutral). A future control query strongly geo-aligned with
        # frame 1 should make it survive over frame 0 at budget 1.
        hist_geo_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        ctrl_geo_matrix = np.array([[0.0, 1.0], [0.0, 0.9], [0.0, 0.8]])
        dino = dino_features({0: [1.0, 0.0], 1: [0.0, 1.0]})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            budget=1,
            hist_query_frame_indices=[0, 1],
            hist_geo_matrix=hist_geo_matrix,
            dino_features=dino,
            ctrl_query_frame_indices=[10, 11, 12],
            ctrl_geo_matrix=ctrl_geo_matrix,
            lambda_hist=0.2,
            return_details=True,
        )
        self.assertTrue(details[1]["mce_selected"])
        self.assertFalse(details[0]["mce_selected"])
        self.assertGreater(details[1]["mce_num_ctrl_queries"], 0)

    def test_no_future_queries_uses_lambda_one(self):
        hist_geo_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        dino = dino_features({0: [1.0, 0.0], 1: [0.0, 1.0]})

        _, details = MEMORY_POLICIES.compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            budget=2,
            hist_query_frame_indices=[0, 1],
            hist_geo_matrix=hist_geo_matrix,
            dino_features=dino,
            return_details=True,
        )
        self.assertEqual(details[0]["mce_lambda"], 1.0)
        self.assertEqual(details[0]["mce_num_ctrl_queries"], 0)

    def test_historical_query_medoids_one_per_cluster(self):
        # Two near-identical views and one distinct view -> two medoids, not
        # three: revisiting a region doesn't add a query for it.
        dino = dino_features({0: [1.0, 0.0], 1: [0.99, 0.01], 2: [0.0, 1.0]})
        medoids, clusters = MEMORY_POLICIES.historical_query_medoids([0, 1, 2], dino)
        self.assertEqual(len(medoids), 2)
        self.assertEqual(len(clusters), 2)


if __name__ == "__main__":
    unittest.main()
