import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_vmem_pairing import compare_runs

try:
    import numpy as np
    from PIL import Image, PngImagePlugin
except ImportError:
    np = Image = PngImagePlugin = None


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

    def test_clip_profile_and_debug_mode_are_not_silently_paired(self):
        self.edit("run_spec.json", lambda spec: spec["arguments"].update(clip_attention="native", generation_debug=None))
        self.assertEqual(compare_runs(self.left, self.right)["provenance_mismatches"], [])
        self.edit("run_spec.json", lambda spec: spec["arguments"].update(clip_attention="math", generation_debug="observe"))
        self.assertEqual(compare_runs(self.left, self.right)["provenance_mismatches"],
                         ["arguments.clip_attention", "arguments.generation_debug"])

    def test_cut3r_profile_is_not_silently_paired(self):
        self.edit("run_spec.json", lambda spec: spec["arguments"].update(cut3r_attention="native"))
        self.assertEqual(compare_runs(self.left, self.right)["provenance_mismatches"], [])
        self.edit("run_spec.json", lambda spec: spec["arguments"].update(cut3r_attention="math"))
        self.assertEqual(compare_runs(self.left, self.right)["provenance_mismatches"],
                         ["arguments.cut3r_attention"])

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


@unittest.skipIf(Image is None, "Pixel checks require Pillow and NumPy")
class PixelPairingAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.left, self.right = fixture(Path(self.tmp.name))
        for path in (self.left, self.right):
            (path / "generated_frames").mkdir()
            for index in range(13):
                self.write_frame(path, index)

    def write_frame(self, path, index, *, pixel=None, metadata=None, size=(3, 2), mode="RGB"):
        frame = path / "generated_frames" / f"{index:04d}.png"
        with Image.new(mode, size) as image:
            if pixel is not None:
                image.putpixel((0, 0), pixel)
            image.save(frame, pnginfo=metadata)
        manifest = path / "frame_manifest.json"
        data = json.loads(manifest.read_text())
        data["sha256"][index] = hashlib.sha256(frame.read_bytes()).hexdigest()
        write_json(manifest, data)

    def report(self):
        return compare_runs(self.left, self.right, compare_pixels=True)

    def test_metadata_only_png_difference_is_not_a_pixel_difference(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("test", "Different PNG metadata, identical RGB")
        self.write_frame(self.right, 1, metadata=metadata)
        report = self.report()
        self.assertIn("saved_frame_hashes", report["pre_eviction_divergences"])
        self.assertNotIn("saved_frame_pixels", report["pre_eviction_divergences"])
        self.assertTrue(report["saved_frame_pixels"]["prefix_pixels_equal"])
        self.assertIsNone(report["saved_frame_pixels"]["first_difference"])
        self.assertFalse(report["saved_frame_pixels"]["frames"][1]["encoded_bytes_equal"])

    def test_first_generated_pixel_difference_and_error_units(self):
        self.write_frame(self.right, 1, pixel=(255, 0, 0))
        report = self.report()
        self.assertIn("saved_frame_pixels", report["pre_eviction_divergences"])
        pixels = report["saved_frame_pixels"]
        self.assertTrue(pixels["file_hashes_verified"])
        self.assertEqual(pixels["frames_compared"], 9)
        row = pixels["first_difference"]
        self.assertEqual(row["index"], 1)
        self.assertEqual(row["max_abs_channel_error_8bit"], 255)
        self.assertAlmostEqual(row["mean_abs_channel_error_8bit"], 255 / 18)
        self.assertAlmostEqual(row["rmse_channel_error_8bit"], 255 / (18 ** 0.5))
        self.assertAlmostEqual(row["changed_pixel_fraction"], 1 / 6)

    def test_frames_generated_by_first_evicting_action_are_compared(self):
        self.write_frame(self.right, 8, pixel=(1, 0, 0))
        self.assertEqual(self.report()["saved_frame_pixels"]["first_difference"]["index"], 8)

    def test_post_eviction_frames_are_not_decoded(self):
        (self.right / "generated_frames" / "0009.png").unlink()
        pixels = self.report()["saved_frame_pixels"]
        self.assertEqual(pixels["frames_compared"], 9)
        self.assertTrue(pixels["prefix_pixels_equal"])

    def test_no_evictions_compares_entire_run(self):
        for filename in ("retrieval_trace.json", "memory_trace.json", "resource_trace.jsonl"):
            (self.right / filename).write_text((self.left / filename).read_text())
        self.write_frame(self.right, 12, pixel=(1, 0, 0))
        pixels = self.report()["saved_frame_pixels"]
        self.assertEqual(pixels["frames_compared"], 13)
        self.assertEqual(pixels["first_difference"]["index"], 12)

    def test_changed_file_with_stale_hash_is_rejected(self):
        frame = self.right / "generated_frames" / "0001.png"
        frame.write_bytes(frame.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.report()

    def test_missing_frame_is_not_a_success(self):
        (self.right / "generated_frames" / "0001.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.report()

    def test_size_mismatch_does_not_resize(self):
        self.write_frame(self.right, 1, size=(4, 2))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.report()

    def test_non_rgb_is_not_silently_converted(self):
        self.write_frame(self.right, 1, mode="RGBA")
        with self.assertRaisesRegex(ValueError, "Expected an RGB PNG"):
            self.report()

    def test_missing_manifest_is_not_a_verified_pixel_comparison(self):
        (self.right / "frame_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires both frame_manifest"):
            self.report()

    def test_pixel_comparison_preserves_all_files(self):
        before = {str(path): path.read_bytes() for path in Path(self.tmp.name).rglob("*") if path.is_file()}
        self.report()
        after = {str(path): path.read_bytes() for path in Path(self.tmp.name).rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
