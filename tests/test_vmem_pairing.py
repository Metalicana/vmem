import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_vmem_pairing import compare_runs


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    paths = [root / "unbounded", root / "bounded"]
    for bounded, path in enumerate(paths):
        path.mkdir()
        args = dict(seed=501, fps=13, num_actions=3, trajectory="pan_45", pattern="forward",
                    step_size=0.1, frames_per_action=4, memory_scope="surfel_indexed_view_memory",
                    frame_storage="resident", inference_steps=50, surfel_niter=400,
                    surfel_reconstruction_window=None, visualize_intermediates=False,
                    memory_policy="slam_covisibility" if bounded else "unbounded",
                    memory_budget=8 if bounded else None)
        write_json(path / "run_spec.json", {
            "arguments": args, "torch_version": "test", "cuda_version": "test",
            "provenance": {"image_sha256": "image", "config_sha256": "config",
                           "source_sha256": {"pipeline": "source"}, "checkpoint_sha256": {"model": "weights"}},
        })
        actions, retrieval, resources = [], [], []
        for step in range(3):
            eligible = list(range(1 + 4 * step))
            context = eligible[-4:]
            surfels = 100 * step
            if bounded and step == 2:
                eligible.remove(1)
                context = [0, 2, 6, 8]
                surfels -= 10
            actions.append(dict(action_index=step, action="right5", current_pose=[[step]],
                                num_generated_frames=4, total_frames_after=5 + step * 4))
            retrieval.append(dict(global_step=step, selected_context_indices=context,
                                  allowed_memory_indices=eligible, num_retained_surfels=surfels,
                                  fallback_used=False))
            resources.append(dict(event="step", step=step,
                                  reconstruction_input_indices=eligible + list(range(1 + 4 * step, 5 + 4 * step)),
                                  after_update={"surfel_count": 100 * (step + 1) - (10 if bounded and step >= 1 else 0),
                                                "appearance_descriptor_sources": {"clip_image_encoder": len(eligible)} if bounded else {},
                                                "frame_payloads": {"resident_counts": {"pil_frames": min(8, 5 + step * 4) if bounded else 5 + step * 4}}}))
        write_json(path / "actions.json", actions)
        write_json(path / "retrieval_trace.json", retrieval)
        (path / "resource_trace.jsonl").write_text("\n".join(json.dumps(row) for row in resources) + "\n")
        write_json(path / "memory_trace.json", [dict(event="memory_eviction", global_step=1,
                   section_end_frame=8, evicted_memory_frame=1)] if bounded else [])
        hashes = [f"{i:064x}" for i in range(13)]
        if bounded:
            hashes[9] = "a" * 64
        write_json(path / "frame_manifest.json", {"schema": "vmem_output_frames_v1", "sha256": hashes})
    return paths


class PairingAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.left, self.right = fixture(Path(self.tmp.name))

    def edit(self, filename, mutate):
        path = self.right / filename
        value = json.loads(path.read_text())
        mutate(value)
        write_json(path, value)

    def test_post_eviction_changes_are_not_reported_as_pre_eviction(self):
        report = compare_runs(self.left, self.right)
        self.assertEqual(report["pre_eviction_divergences"], [])
        self.assertEqual(report["first_differences"]["geometry_after_update"]["index"], 1)
        self.assertEqual(report["first_differences"]["reconstruction_inputs"]["index"], 2)
        self.assertTrue(report["saved_frames"]["prefix_hashes_equal"])
        self.assertEqual(report["saved_frames"]["first_difference"]["index"], 9)
        self.assertEqual(report["provenance_mismatches"], [])
        self.assertEqual(report["missing_provenance"], [])

    def test_generation_on_first_evicting_action_is_still_before_eviction(self):
        self.edit("retrieval_trace.json", lambda rows: rows[1].update(selected_context_indices=[0, 1, 2, 3]))
        self.edit("frame_manifest.json", lambda data: data["sha256"].__setitem__(8, "b" * 64))
        report = compare_runs(self.left, self.right)
        self.assertIn("context", report["pre_eviction_divergences"])
        self.assertIn("saved_frame_hashes", report["pre_eviction_divergences"])
        self.assertEqual(report["saved_frames"]["frames_compared_before_first_eviction"], 9)

    def test_reports_pre_eviction_geometry_difference(self):
        self.edit("retrieval_trace.json", lambda rows: rows[1].update(num_retained_surfels=99))
        report = compare_runs(self.left, self.right)
        self.assertIn("geometry_at_retrieval", report["pre_eviction_divergences"])

    def test_missing_hashes_are_unavailable_not_equal(self):
        (self.right / "frame_manifest.json").unlink()
        report = compare_runs(self.left, self.right)
        self.assertEqual(report["saved_frames"]["status"], "unavailable")
        self.assertNotIn("prefix_hashes_equal", report["saved_frames"])

    def test_flags_source_and_command_mismatch(self):
        self.edit("run_spec.json", lambda spec: spec["provenance"]["source_sha256"].update(pipeline="different"))
        self.edit("actions.json", lambda rows: rows[0].update(action="left5"))
        report = compare_runs(self.left, self.right)
        self.assertIn("provenance.source_sha256", report["provenance_mismatches"])
        self.assertEqual(report["first_differences"]["actions"]["index"], 0)

    def test_missing_provenance_is_not_a_verified_match(self):
        self.edit("run_spec.json", lambda spec: spec["provenance"].pop("checkpoint_sha256"))
        self.assertIn("provenance.checkpoint_sha256", compare_runs(self.left, self.right)["missing_provenance"])

    def test_trace_alignment_and_completeness_are_required(self):
        self.edit("retrieval_trace.json", lambda rows: rows[1].update(global_step=0))
        with self.assertRaisesRegex(ValueError, "unordered global_step"):
            compare_runs(self.left, self.right)
        self.edit("retrieval_trace.json", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "complete"):
            compare_runs(self.left, self.right)

    def test_hash_manifest_must_cover_entire_video(self):
        self.edit("frame_manifest.json", lambda data: data["sha256"].pop())
        with self.assertRaisesRegex(ValueError, "frame manifest"):
            compare_runs(self.left, self.right)

    def test_no_evictions_means_entire_run_is_a_noop_control(self):
        for filename in ("retrieval_trace.json", "memory_trace.json", "resource_trace.jsonl", "frame_manifest.json"):
            (self.right / filename).write_text((self.left / filename).read_text())
        self.edit("frame_manifest.json", lambda data: data["sha256"].__setitem__(12, "a" * 64))
        report = compare_runs(self.left, self.right)
        self.assertIsNone(report["first_eviction"])
        self.assertIn("saved_frame_hashes", report["pre_eviction_divergences"])
        self.assertEqual(report["saved_frames"]["frames_compared_before_first_eviction"], 13)

    def test_does_not_modify_run_records(self):
        before = {str(path): path.read_bytes() for path in Path(self.tmp.name).rglob("*") if path.is_file()}
        compare_runs(self.left, self.right)
        after = {str(path): path.read_bytes() for path in Path(self.tmp.name).rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
