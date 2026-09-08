import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import vmem_recovery as recovery
from scripts.run_vmem_demo_manifest import _command_for_row, _selected_indices, run_logged
from scripts.vmem_protocol import expected_settings, verify_lock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_test_policies", ROOT / "modeling" / "memory_policies.py")
POLICIES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICIES
SPEC.loader.exec_module(POLICIES)


def fixture():
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    buffer = POLICIES.FrameMemoryBuffer("slam_covisibility", 2, pinned_frames={0})
    buffer.update([0])
    pipeline = SimpleNamespace(
        device="cpu", pil_frames=[image], latents=[np.zeros((2, 3), dtype=np.float32)],
        encoder_embeddings=[np.ones(3, dtype=np.float32)], c2ws=[np.eye(4)], Ks=[np.eye(3)],
        poses=[], focal_lengths=[], surfel_Ks=[1.0], surfel_depths=[np.ones((4, 4), dtype=np.float32)],
        surfels=[SimpleNamespace(position=np.zeros(3), normal=np.ones(3), radius=1.0)],
        surfel_to_timestep={0: [0]}, memory_policy="slam_covisibility", memory_budget=2,
        memory_scope="surfel_indexed_view_memory", surfel_reconstruction_window=None,
        memory_buffer=buffer, global_step=0, initial_threshold=0.3, retrieval_trace=[], memory_events=[],
        model="weights must never enter the checkpoint",
    )
    navigator = SimpleNamespace(current_pose=np.eye(4), current_K=np.eye(3), pose_history=[], frames=[image])
    return pipeline, navigator


class RecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is required for recovery round-trip tests")
        cls.torch = torch

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.arguments = expected_settings({"image": "test.jpg", "run_id": "test", "num_actions": 3,
                                           "memory_policy": "slam_covisibility", "memory_budget": 2})
        self.run_identity = recovery.identity(self.arguments, {"source_sha256": {"source": "hash"}}, self.torch)
        self.pipeline, self.navigator = fixture()
        self.records, self.hashes = [], []

    def advance(self, pipeline, navigator, records):
        value = random.random() + float(np.random.random()) + float(self.torch.rand(()))
        first = len(pipeline.pil_frames)
        for index in range(first, first + 4):
            pipeline.pil_frames.append(Image.new("RGB", (8, 8), (int(value * 60), index, 20)))
            pipeline.latents.append(np.full((2, 3), value, dtype=np.float32))
            pipeline.encoder_embeddings.append(np.ones(3, dtype=np.float32) * value)
            pipeline.c2ws.append(np.eye(4))
            pipeline.Ks.append(np.eye(3))
            pipeline.surfel_depths.append(np.ones((4, 4), dtype=np.float32))
            pipeline.surfel_Ks.append(1.0)
        pipeline.global_step += 1
        evicted = pipeline.memory_buffer.update(range(first, first + 4), protected_frames={first + 3})
        pipeline.memory_events.extend(evicted)
        pipeline.surfel_to_timestep = {0: pipeline.memory_buffer.candidates()}
        pipeline.retrieval_trace.append({"global_step": pipeline.global_step - 1, "random_value": value})
        navigator.frames.extend(pipeline.pil_frames[first:])
        navigator.pose_history.append({"value": value})
        records.append({"action_index": len(records), "random_value": value})
        return value

    def commit(self):
        recovery.save_new_frames(self.source, self.pipeline.pil_frames, self.hashes)
        return recovery.save_checkpoint(self.source, self.pipeline, self.navigator, self.records,
                                        self.hashes, self.run_identity, self.torch)

    def test_round_trip_preserves_next_generation_rng_bank_geometry_and_aliases(self):
        random.seed(5)
        np.random.seed(5)
        self.torch.manual_seed(5)
        self.advance(self.pipeline, self.navigator, self.records)
        # Preserve the duplicate initial-frame alias used by the real Navigator.
        self.navigator.frames.append(self.pipeline.pil_frames[0])
        self.pipeline.surfel_Ks[0] = self.torch.tensor(2.0)
        self.commit()
        expected = self.advance(self.pipeline, self.navigator, self.records)
        state, sha = recovery.load_checkpoint(self.source, self.run_identity, self.torch, "cpu")
        self.assertEqual(len(sha), 64)
        self.assertNotIn("model", state["pipeline"])
        restored, navigator = fixture()
        target = self.root / "resumed"
        target.mkdir()
        records, hashes, rng = recovery.restore_into_new_attempt(self.source, target, state, restored, navigator)
        self.assertIs(navigator.frames[0], restored.pil_frames[0])
        self.assertIs(navigator.frames[-1], restored.pil_frames[0])
        self.assertEqual(restored.initial_threshold, 0.3)
        self.assertEqual(float(restored.surfel_Ks[0]), 2.0)
        self.assertEqual(restored.memory_buffer.candidates(), [0, 4])
        self.assertEqual(len(hashes), 5)
        self.torch.rand(10)
        np.random.random(10)
        random.random()
        recovery.restore_rng(rng, self.torch, "cpu")
        actual = self.advance(restored, navigator, records)
        self.assertEqual(expected, actual)
        self.assertEqual(restored.memory_buffer.candidates(), self.pipeline.memory_buffer.candidates())
        self.assertEqual(restored.surfel_to_timestep, self.pipeline.surfel_to_timestep)
        self.assertEqual(records, self.records)
        np.testing.assert_array_equal(restored.latents[-1], self.pipeline.latents[-1])

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        self.commit()
        path = self.source / "recovery" / "latest.pt"
        before = recovery.digest_file(path)
        def interrupted(handle):
            handle.write(b"incomplete")
            raise OSError("simulated disk failure")
        with self.assertRaises(OSError):
            recovery.atomic_write(path, interrupted)
        self.assertEqual(recovery.digest_file(path), before)
        self.assertFalse(list(path.parent.glob("*.tmp-*")))

    def test_trace_recovery_ignores_later_incomplete_tail_without_modifying_parent(self):
        trace = self.source / "resource_trace.jsonl"
        prefix = '{"event":"schema"}\n'
        trace.write_text(prefix)
        self.commit()
        trace.write_text(prefix + '{"event":"step", "incomplete":')
        before = trace.read_bytes()
        state, _ = recovery.load_checkpoint(self.source, self.run_identity, self.torch, "cpu")
        target = self.root / "resumed"
        target.mkdir()
        pipeline, navigator = fixture()
        recovery.restore_into_new_attempt(self.source, target, state, pipeline, navigator)
        self.assertEqual((target / trace.name).read_text(), prefix)
        self.assertEqual(trace.read_bytes(), before)
        self.assertNotEqual((target / "generated_frames/0000.png").stat().st_ino,
                            (self.source / "generated_frames/0000.png").stat().st_ino)

    def test_changed_input_settings_or_weights_rejected(self):
        self.commit()
        for category, key in (("arguments", "seed"), ("provenance", "image_sha256"),
                              ("provenance", "checkpoint_sha256"), ("provenance", "source_sha256")):
            wrong = copy.deepcopy(self.run_identity)
            wrong[category][key] = "changed"
            with self.assertRaises(ValueError):
                recovery.load_checkpoint(self.source, wrong, self.torch, "cpu")

    def test_missing_checkpoint_and_damaged_frame_rejected(self):
        with self.assertRaises(FileNotFoundError):
            recovery.load_checkpoint(self.source, self.run_identity, self.torch, "cpu")
        self.commit()
        (self.source / "generated_frames/0000.png").write_bytes(b"damaged")
        state, _ = recovery.load_checkpoint(self.source, self.run_identity, self.torch, "cpu")
        target = self.root / "resumed"
        target.mkdir()
        pipeline, navigator = fixture()
        with self.assertRaises(ValueError):
            recovery.restore_into_new_attempt(self.source, target, state, pipeline, navigator)

    def test_durable_frames_cannot_be_overwritten(self):
        recovery.save_new_frames(self.source, self.pipeline.pil_frames, self.hashes)
        with self.assertRaises(FileExistsError):
            recovery.save_new_frames(self.source, self.pipeline.pil_frames, [])


class RecoveryLauncherTest(unittest.TestCase):
    def test_selected_pair_and_resume_flag(self):
        args = SimpleNamespace(job_indices=[0, 1], all=False, job_index=None)
        self.assertEqual(_selected_indices(args, [{}, {}, {}]), [0, 1])
        command = _command_for_row({"image": "test.jpg"}, output_root=Path("outputs"),
                                   config=Path("config.yaml"), device="cuda", dry_run=False,
                                   resume_from=Path("old_run"))
        self.assertEqual(command[-2:], ["--resume-from", "old_run"])
        args.job_indices = [0, 0]
        with self.assertRaises(ValueError):
            _selected_indices(args, [{}, {}, {}])

    def test_launcher_persists_exit_signal_and_output(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            def start(command, **kwargs):
                kwargs["stdout"].write("test child output\n")
                self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
                return SimpleNamespace(pid=123, wait=lambda: -9, poll=lambda: -9)
            with patch("scripts.run_vmem_demo_manifest.subprocess.Popen", side_effect=start):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_logged(["fake-runner"], log_dir=Path(tmp), index=0)
            status = json.loads(next(Path(tmp).glob("*.json")).read_text())
            self.assertEqual(status["returncode"], -9)
            self.assertEqual(status["pid"], 123)
            self.assertEqual(next(Path(tmp).glob("*.log")).read_text(), "test child output\n")

    def test_checkpoint_interval_is_frozen_in_new_locks(self):
        row = {"image": "test.jpg", "run_id": "test"}
        settings = expected_settings(row)
        provenance = {"source_sha256": {}, "config_sha256": "config", "image_sha256": "image"}
        lock = {"schema": "vmem_experiment_lock_v2", "rows": [row], "image_sha256": {"test.jpg": "image"},
                "source_sha256": {}, "config_sha256": "config"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            path.write_text(json.dumps(lock))
            verify_lock(path, settings, provenance)
            with self.assertRaises(ValueError):
                verify_lock(path, {**settings, "checkpoint_every": 1}, provenance)


if __name__ == "__main__":
    unittest.main()
