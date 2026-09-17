import copy
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import evaluate_vmem_quality as quality
from scripts import run_vmem_vbench_long as adapter


def inventory():
    video = {"frames": 781, "width": 576, "height": 576, "fps": 13.0, "seconds": 60.076923}
    arms = [{"status": "validated", "policy": policy, "run_dir": f"outputs/{policy}", "video": video.copy()}
            for policy in quality.POLICIES]
    return {"schema": "vmem_inventory_v1", "pairs": [
        {"case_id": "oxford_pan_45", "validated_pair": True, "arms": arms},
        {"case_id": "missing", "validated_pair": False, "arms": [None, None]},
    ]}


class QualitySelectionTest(unittest.TestCase):
    def test_selects_only_inventory_matched_pair(self):
        selected = quality.select_pairs(inventory())
        self.assertEqual([row["policy"] for row in selected], list(quality.POLICIES))
        self.assertEqual({row["case_id"] for row in selected}, {"oxford_pan_45"})

    def test_rejects_incomplete_requested_case_and_bad_pairs(self):
        for case in ("missing", "absent"):
            with self.subTest(case=case), self.assertRaises(ValueError):
                quality.select_pairs(inventory(), case)
        variants = []
        duplicate = inventory()
        duplicate["pairs"][0]["arms"][1]["policy"] = "unbounded"
        variants.append(duplicate)
        wrong_format = inventory()
        wrong_format["pairs"][0]["arms"][1]["video"]["fps"] = 24.0
        variants.append(wrong_format)
        unsafe = inventory()
        unsafe["pairs"][0]["case_id"] = "../escape"
        variants.append(unsafe)
        repeated = inventory()
        repeated["pairs"].append(copy.deepcopy(repeated["pairs"][0]))
        variants.append(repeated)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                quality.select_pairs(payload)

    def test_781_frames_cover_full_video_with_declared_overlapping_tail(self):
        plan = quality.clip_plan({"frames": 781, "fps": 13.0})
        self.assertEqual(len(plan), 31)
        self.assertEqual(plan[-1]["start_frame"], 755)
        self.assertEqual(plan[-1]["end_frame_exclusive"], 781)
        self.assertTrue(plan[-1]["overlapping_tail"])
        self.assertEqual(sum(row["overlapping_tail"] for row in plan), 1)
        covered = {frame for row in plan for frame in range(row["start_frame"], row["end_frame_exclusive"])}
        self.assertEqual(covered, set(range(781)))
        self.assertEqual(sum(row["end_frame_exclusive"] - row["start_frame"] for row in plan) - len(covered), 25)

    def test_exact_60_seconds_has_no_overlapping_tail(self):
        plan = quality.clip_plan({"frames": 780, "fps": 13.0})
        self.assertEqual(len(plan), 30)
        self.assertFalse(any(row["overlapping_tail"] for row in plan))

    def test_rejects_fractional_fps_and_too_short_video(self):
        for video in ({"frames": 781, "fps": 12.99}, {"frames": 1, "fps": 13.0}):
            with self.subTest(video=video), self.assertRaises(ValueError):
                quality.clip_plan(video)

    def test_verify_clips_checks_length_format_and_missing_files(self):
        video = {"frames": 53, "fps": 13.0, "width": 576, "height": 576}
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            clips = stage / "split_clip" / "video-0"
            clips.mkdir(parents=True)
            paths = [clips / f"video-0_{index:03d}.mp4" for index in range(3)]
            for path in paths:
                path.write_bytes(b"clip")
            with patch.object(quality, "probe_video", return_value={**video, "frames": 26}):
                self.assertEqual(len(quality.verify_clips(stage, video)), 3)
            with patch.object(quality, "probe_video", return_value={**video, "frames": 25}):
                with self.assertRaisesRegex(ValueError, "format/length"):
                    quality.verify_clips(stage, video)
            paths[-1].unlink()
            with self.assertRaisesRegex(ValueError, "coverage"):
                quality.verify_clips(stage, video)


class QualityGroupingTest(unittest.TestCase):
    def rows(self, stage, stem, scores):
        (stage / f"{stem}.mp4").write_bytes(b"video")
        clips = stage / "split_clip" / stem
        clips.mkdir(parents=True)
        return [{"video_path": str(clips / f"{stem}_{index:03d}.mp4"), "video_results": score}
                for index, score in enumerate(scores)]

    def test_underscore_names_do_not_merge_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(Path(tmp), "video_a", [0.2, 0.4])
            rows += self.rows(Path(tmp), "video_b", [0.6, 0.8])
            mean, details, videos = adapter.reorganize_clips_results(rows)
            self.assertAlmostEqual(mean, 0.5)
            self.assertEqual(details, rows)
            self.assertEqual(len(videos), 2)

    def test_imaging_top_level_normalization_matches_sibling_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(Path(tmp), "video-0", [40.0, 60.0])
            mean, _, videos = adapter.reorganize_clips_results(rows, "imaging_quality")
            self.assertAlmostEqual(mean, 0.5)
            self.assertAlmostEqual(videos[0]["video_results"], 50.0)

    def test_rejects_duplicate_or_nan_clip_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(Path(tmp), "video-0", [0.5])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                adapter.reorganize_clips_results(rows + rows)
            rows[0]["video_results"] = float("nan")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                adapter.reorganize_clips_results(rows)


class QualityResultsTest(unittest.TestCase):
    def result(self, base, dimension="imaging_quality", score=0.5):
        base.mkdir(parents=True, exist_ok=True)
        staged = base / "input" / "video-0.mp4"
        plan = quality.clip_plan({"frames": 52, "fps": 13.0})
        for row in plan:
            row["path"] = str(staged.parent / "split_clip/video-0" / f"video-0_{row['clip_index']:03d}.mp4")
        scale = 100 if dimension == "imaging_quality" else 1
        details = [{"video_path": row["path"], "video_results": score * scale} for row in plan]
        videos = [{"video_path": str(staged), "video_results": score * scale}]
        result = [score, videos] if dimension in quality.CONSISTENCY else [score, details, videos]
        path = base / "results_eval_results.json"
        quality.write_json(path, {dimension: result})
        return path, staged, plan

    def test_normalizes_imaging_clip_scores_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, staged, plan = self.result(Path(tmp))
            score, clips = quality.parse_result(path, "imaging_quality", staged, plan)
            self.assertEqual(score, 0.5)
            self.assertEqual([row["score"] for row in clips], [0.5, 0.5])

    def test_reads_fused_consistency_without_inventing_clip_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, staged, plan = self.result(Path(tmp), "subject_consistency", 0.7)
            score, clips = quality.parse_result(path, "subject_consistency", staged, plan)
            self.assertEqual(score, 0.7)
            self.assertEqual(clips, [])

    def test_full_info_cannot_omit_or_duplicate_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, plan = self.result(Path(tmp))
            path = Path(tmp) / "full_info.json"
            clips = [row["path"] for row in plan]
            for selection in (clips, clips[:-1], [clips[0], clips[0]]):
                quality.write_json(path, [{"dimension": ["imaging_quality"], "video_list": selection}])
                if selection == clips:
                    quality.verify_full_info(path, "imaging_quality", plan)
                else:
                    with self.assertRaisesRegex(ValueError, "every expected clip"):
                        quality.verify_full_info(path, "imaging_quality", plan)

    def test_invalid_evaluator_provenance_blocks_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            quality.write_json(output / "invalid.json", {"reason": "Evaluator source changed"})
            with self.assertRaisesRegex(ValueError, "source changed"):
                quality.summarize(output, display=False)

    def test_rejects_missing_duplicate_wrong_video_and_wrong_scale_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, staged, plan = self.result(Path(tmp))
            original = quality.read_json(path)
            variants = []
            missing = copy.deepcopy(original)
            missing["imaging_quality"][1].pop()
            variants.append(missing)
            duplicate = copy.deepcopy(original)
            duplicate["imaging_quality"][1].append(duplicate["imaging_quality"][1][0])
            variants.append(duplicate)
            wrong_video = copy.deepcopy(original)
            wrong_video["imaging_quality"][2][0]["video_path"] = str(Path(tmp) / "other.mp4")
            variants.append(wrong_video)
            wrong_scale = copy.deepcopy(original)
            wrong_scale["imaging_quality"][0] = 50.0
            variants.append(wrong_scale)
            for payload in variants:
                quality.write_json(path, payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    quality.parse_result(path, "imaging_quality", staged, plan)

    def test_summary_preserves_missing_arm_then_reports_paired_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            inputs, jobs = [], []
            for index, (policy, score) in enumerate(zip(quality.POLICIES, (0.6, 0.5))):
                base = output / "oxford" / policy
                path, staged, plan = self.result(base, score=score)
                quality.write_json(base / "clip_coverage.json", plan)
                inputs.append({"case_id": "oxford", "policy": policy, "staged_video": str(staged)})
                jobs.append({"input_index": index, "dimension": "imaging_quality", "status": "pending",
                             "result": str(path), "result_sha256": quality.sha256(path)})
            quality.write_json(output / "evaluation_spec.json", {"inputs": inputs, "dimensions": ["imaging_quality"]})
            jobs[0]["status"] = "complete"
            quality.write_json(output / "jobs.json", jobs)
            rows = quality.summarize(output, display=False)
            self.assertIsNone(rows[0]["geocov_minus_unbounded"])
            self.assertEqual(rows[0]["status"], "incomplete")
            jobs[1]["status"] = "complete"
            quality.write_json(output / "jobs.json", jobs)
            rows = quality.summarize(output, display=False)
            self.assertAlmostEqual(rows[0]["geocov_minus_unbounded"], -0.1)
            self.assertEqual(rows[0]["status"], "complete")
            Path(jobs[0]["result"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                quality.summarize(output, display=False)


class QualityRunnerTest(unittest.TestCase):
    def test_failed_log_prints_tail_instead_of_generic_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluate.log"
            path.write_text("early line\n" + "progress\n" * 100 + "ModuleNotFoundError: missing dependency\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                quality.report_failed_log(path)
            self.assertIn("ModuleNotFoundError: missing dependency", stderr.getvalue())
            self.assertNotIn("early line", stderr.getvalue())
            self.assertIn(str(path), stderr.getvalue())

    def test_missing_log_does_not_mask_original_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                quality.report_failed_log(Path(tmp) / "absent.log")
            self.assertIn("Could not read evaluator log", stderr.getvalue())

    def test_failed_log_read_is_bounded_and_tolerates_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluate.log"
            path.write_bytes(b"x" * 32768 + b"\xff\nRuntimeError: evaluator failed\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                quality.report_failed_log(path)
            self.assertIn("RuntimeError: evaluator failed", stderr.getvalue())
            self.assertLess(len(stderr.getvalue()), 17000)

    def test_mocked_runner_stages_unchanged_pair_and_writes_signed_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = inventory()
            original_paths = []
            video = {"frames": 52, "width": 576, "height": 576, "fps": 13.0, "seconds": 4.0}
            for arm in payload["pairs"][0]["arms"]:
                run_dir = root / "generation" / arm["policy"]
                run_dir.mkdir(parents=True)
                arm.update(run_dir=str(run_dir), video=video.copy())
                (run_dir / "generated.mp4").write_bytes(b"original")
                quality.write_json(run_dir / "run_spec.json", {"provenance": {"test": True}})
                original_paths.append(run_dir / "generated.mp4")
            source = root / "inventory.json"
            quality.write_json(source, payload)
            args = SimpleNamespace(inventory=source, vbench_root=root / "VBench", output=root / "quality",
                                   case_id="oxford_pan_45", dimensions=["aesthetic_quality", "imaging_quality"])

            def fake_worker(command, **kwargs):
                if "--check" in command:
                    report = Path(command[command.index("--check-report") + 1])
                    quality.write_json(report, {"status": "passed", "dimensions": args.dimensions})
                    return
                stage = Path(command[command.index("--videos_path") + 1])
                dimension = command[command.index("--dimension") + 1]
                result_dir = Path(command[command.index("--output_path") + 1])
                clip_dir = stage / "split_clip/video-0"
                clip_dir.mkdir(parents=True, exist_ok=True)
                paths = [clip_dir / f"video-0_{index:03d}.mp4" for index in range(2)]
                for path in paths:
                    path.write_bytes(b"clip")
                score = 0.6 if stage.parent.name == "unbounded" else 0.5
                scale = 100 if dimension == "imaging_quality" else 1
                quality.write_json(result_dir / "test_eval_results.json", {dimension: [score,
                    [{"video_path": str(path), "video_results": score * scale} for path in paths],
                    [{"video_path": str(stage / "video-0.mp4"), "video_results": score * scale}]]})
                quality.write_json(result_dir / "test_full_info.json", [
                    {"dimension": [dimension], "video_list": [str(path) for path in paths]}])

            def fake_probe(path):
                return video if Path(path).name == "generated.mp4" else {**video, "frames": 26, "seconds": 2.0}

            with patch.object(quality, "vbench_sources", return_value={"source": "hash"}), \
                    patch.object(quality, "probe_video", side_effect=fake_probe), \
                    patch.object(quality.shutil, "which", return_value="/bin/tool"), \
                    patch.object(quality.subprocess, "run", side_effect=fake_worker) as worker:
                quality.run(args)
                self.assertEqual(worker.call_count, 5)
                with self.assertRaises(FileExistsError):
                    quality.run(args)
                self.assertEqual(worker.call_count, 5)
            summary = quality.read_json(args.output / "summary.json")
            self.assertEqual(summary["completed_jobs"], 4)
            for row in summary["paired"]:
                self.assertAlmostEqual(row["geocov_minus_unbounded"], -0.1)
            for path in original_paths:
                self.assertEqual(path.read_bytes(), b"original")
                self.assertFalse((path.parent / "split_clip").exists())

    def test_failed_preflight_leaves_no_quality_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "inventory.json"
            quality.write_json(source, inventory())
            args = SimpleNamespace(inventory=source, vbench_root=root / "VBench", output=root / "quality",
                                   case_id="oxford_pan_45", dimensions=["aesthetic_quality"])
            with patch.object(quality, "vbench_sources", return_value={"source": "hash"}), \
                    patch.object(quality.shutil, "which", return_value="/bin/tool"), \
                    patch.object(quality.subprocess, "run", side_effect=quality.subprocess.CalledProcessError(1, "preflight")), \
                    self.assertRaises(quality.subprocess.CalledProcessError):
                quality.run(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
