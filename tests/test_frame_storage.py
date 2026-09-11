import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image

from frame_storage import owned_frame_array, payload_summary, release_evicted_payloads, validate_resident_payloads
from scripts import vmem_recovery as recovery
from scripts.vmem_protocol import expected_settings, verify_lock
from scripts.build_vmem_transfer_manifest import build_rows


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("storage_test_policies", ROOT / "modeling/memory_policies.py")
POLICIES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICIES
SPEC.loader.exec_module(POLICIES)


def pipeline_type():
    source = ast.parse((ROOT / "modeling/pipeline.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "VMemPipeline")
    names = {"get_allowed_memory_indices", "_prune_surfels_to_memory", "_update_memory_budget"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"profiled_phase": lambda name: lambda function: function}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "pipeline.py", "exec"), namespace)
    return namespace["VMemPipeline"]


def fixture(mode="resident", policy="slam_covisibility"):
    pipeline = pipeline_type()()
    pipeline.frame_storage = mode
    pipeline.device = "cpu"
    pipeline.memory_policy, pipeline.memory_scope = policy, "surfel_indexed_view_memory"
    pipeline.memory_budget = None if policy == "unbounded" else 32
    pipeline.memory_buffer = None if policy == "unbounded" else POLICIES.FrameMemoryBuffer(policy, 32, pinned_frames={0})
    if pipeline.memory_buffer is not None:
        pipeline.memory_buffer.update([0])
    pipeline.pil_frames = [Image.new("RGB", (8, 8), (1, 2, 3))]
    pipeline.latents = [np.zeros((2, 3), dtype=np.float32)]
    pipeline.encoder_embeddings = [np.ones(4, dtype=np.float32)]
    pipeline.Ks, pipeline.c2ws = [np.eye(3)], [np.eye(4)]
    pipeline.surfel_depths, pipeline.surfel_Ks = [], []
    pipeline.surfels, pipeline.surfel_to_timestep = [], {}
    pipeline.memory_events, pipeline.retrieval_trace = [], []
    pipeline._pending_payload_evictions, pipeline._dino_feature_cache = [], {}
    pipeline.durable_frame_count, pipeline.global_step = 0, 0
    pipeline._record_memory_event = lambda event: pipeline.memory_events.append(event)
    pipeline._pinned_memory_frames = lambda: {0}
    def scores(indices):
        pipeline.memory_descriptor_sources = {"clip_image_encoder": len(indices)}
        return POLICIES.compute_slam_covisibility_scores(
            memory_frame_indices=indices, c2ws=np.asarray(pipeline.c2ws), pinned_frames={0},
            latent_features={index: pipeline.encoder_embeddings[index] for index in indices}, return_details=True,
        )
    pipeline._compute_memory_scores = scores
    return pipeline


def advance(pipeline):
    first = len(pipeline.pil_frames)
    for index in range(first, first + 4):
        pipeline.pil_frames.append(Image.new("RGB", (8, 8), (index, 20, 30)))
        pipeline.latents.append(np.full((2, 3), index, dtype=np.float32))
        pipeline.encoder_embeddings.append(np.array([1, index % 7, index % 5, 2], dtype=np.float32))
        pipeline.Ks.append(np.eye(3))
        pose = np.eye(4)
        pose[0, 3] = index % 12 * 0.01
        pipeline.c2ws.append(pose)
    new = list(range(first, first + 4))
    inputs = sorted(set(pipeline.get_allowed_memory_indices()) | set(new))
    depths = np.ones((len(inputs), 4, 4), dtype=np.float32)
    root_ref = weakref.ref(depths)
    while len(pipeline.surfel_depths) < len(pipeline.pil_frames):
        pipeline.surfel_depths.append(None)
        pipeline.surfel_Ks.append(1.0)
    for local, index in enumerate(inputs):
        pipeline.surfel_depths[index] = owned_frame_array(depths[local], pipeline.frame_storage)
    for index in new:
        pipeline.surfel_to_timestep[len(pipeline.surfels)] = [index]
        pipeline.surfels.append(SimpleNamespace(position=np.zeros(3)))
    pipeline._update_memory_budget(new, protected_frames={new[-1]})
    pipeline.global_step += 1
    return root_ref


class FrameStorageTest(unittest.TestCase):
    def test_actual_eviction_releases_payloads_and_depth_roots_beyond_32(self):
        pipeline = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            recovery.save_new_frames(tmp, pipeline.pil_frames, hashes)
            release_evicted_payloads(pipeline, len(hashes))
            plateau = None
            for _ in range(12):
                root = advance(pipeline)
                evicted = list(pipeline._pending_payload_evictions)
                old_images = [weakref.ref(pipeline.pil_frames[index]) for index in evicted]
                recovery.save_new_frames(tmp, pipeline.pil_frames, hashes)
                release_evicted_payloads(pipeline, len(hashes))
                self.assertIsNone(root())
                self.assertTrue(all(reference() is None for reference in old_images))
                summary = validate_resident_payloads(pipeline)
                self.assertEqual(summary["resident_counts"]["pil_frames"], min(32, len(hashes)))
                self.assertTrue(all(ref in pipeline.get_allowed_memory_indices()
                                    for refs in pipeline.surfel_to_timestep.values() for ref in refs))
                if len(hashes) > 32:
                    size = summary["total_logical_bytes"]
                    if plateau is not None:
                        self.assertEqual(size, plateau)
                    plateau = size
            self.assertEqual(len(hashes), 49)
            self.assertEqual(len(list(Path(tmp).glob("generated_frames/*.png"))), 49)
            self.assertEqual(sum(frame is not None for frame in pipeline.pil_frames), 32)
            pixels = [tuple(image.getpixel((0, 0))) for image in recovery.iter_saved_frames(tmp, hashes)]
            self.assertEqual(len(pixels), 49)
            self.assertEqual(pixels[-1], (48, 20, 30))

    def test_storage_change_preserves_scores_bank_and_surfel_pruning(self):
        legacy, resident = fixture("legacy"), fixture()
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            recovery.save_new_frames(tmp, resident.pil_frames, hashes)
            release_evicted_payloads(resident, len(hashes))
            for _ in range(12):
                advance(legacy)
                advance(resident)
                recovery.save_new_frames(tmp, resident.pil_frames, hashes)
                release_evicted_payloads(resident, len(hashes))
                self.assertEqual(resident.get_allowed_memory_indices(), legacy.get_allowed_memory_indices())
                self.assertEqual(resident.memory_events, legacy.memory_events)
                self.assertEqual(resident.surfel_to_timestep, legacy.surfel_to_timestep)
                for index in resident.get_allowed_memory_indices():
                    np.testing.assert_array_equal(resident.surfel_depths[index], legacy.surfel_depths[index])

    def test_disk_failure_does_not_release_unwritten_payloads(self):
        pipeline = fixture()
        advance(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(recovery, "atomic_write", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    recovery.save_new_frames(tmp, pipeline.pil_frames, [])
            with self.assertRaises(ValueError):
                release_evicted_payloads(pipeline, 0)
            self.assertTrue(all(image is not None for image in pipeline.pil_frames))

    def test_owned_arrays_preserve_values_without_pinning_batch_storage(self):
        root = np.arange(128, dtype=np.float32).reshape(8, 16)
        array = owned_frame_array(root[3], "resident")
        np.testing.assert_array_equal(array, root[3])
        self.assertTrue(array.flags.owndata)
        self.assertFalse(np.shares_memory(array, root))

    def test_unbounded_still_retains_every_frame(self):
        pipeline = fixture(policy="unbounded")
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            for _ in range(12):
                advance(pipeline)
                recovery.save_new_frames(tmp, pipeline.pil_frames, hashes)
                release_evicted_payloads(pipeline, len(hashes))
            self.assertEqual(validate_resident_payloads(pipeline)["resident_counts"]["pil_frames"], 49)

    def test_v2_changes_storage_only_and_requires_new_lock(self):
        for old, new in zip(build_rows(), build_rows("v2")):
            self.assertEqual(new["frame_storage"], "resident")
            normalized = {key: value.replace("transfer_v2_", "transfer_v1_") if isinstance(value, str) else value
                          for key, value in new.items() if key != "frame_storage"}
            self.assertEqual(normalized, old)
        row = build_rows("v2")[0]
        settings = expected_settings(row)
        provenance = {"source_sha256": {}, "config_sha256": "config", "image_sha256": "image"}
        lock = {"schema": "vmem_experiment_lock_v3", "rows": [row], "image_sha256": {row["image"]: "image"},
                "source_sha256": {}, "config_sha256": "config"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            path.write_text(json.dumps(lock))
            verify_lock(path, settings, provenance)
            with self.assertRaises(ValueError):
                verify_lock(path, {**settings, "frame_storage": "legacy"}, provenance)
            lock["schema"] = "vmem_experiment_lock_v2"
            path.write_text(json.dumps(lock))
            with self.assertRaises(ValueError):
                verify_lock(path, settings, provenance)


class ResidentRecoveryTest(unittest.TestCase):
    def test_checkpoint_resume_loads_only_retained_images_and_exports_all_frames(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch is required for checkpoint tests")
        pipeline = fixture()
        navigator = SimpleNamespace(frames=[], current_pose=np.eye(4), current_K=np.eye(3), pose_history=[])
        settings = expected_settings({"image": "test.jpg", "run_id": "test", "num_actions": 12,
                                      "frame_storage": "resident", "memory_policy": "slam_covisibility", "memory_budget": 32})
        run_identity = recovery.identity(settings, {}, torch)
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "resumed"
            source.mkdir()
            target.mkdir()
            hashes, records = [], []
            for index in range(10):
                advance(pipeline)
                recovery.save_new_frames(source, pipeline.pil_frames, hashes)
                release_evicted_payloads(pipeline, len(hashes))
                records.append({"action_index": index})
            recovery.save_checkpoint(source, pipeline, navigator, records, hashes, run_identity, torch)
            state, _ = recovery.load_checkpoint(source, run_identity, torch, "cpu")
            self.assertEqual(len(state["resident_frame_indices"]), 32)
            self.assertEqual(sum(value is not None for value in state["pipeline"]["latents"]), 32)
            restored = fixture()
            real_open = Image.open
            with patch.object(recovery.Image, "open", wraps=real_open) as opened:
                restored_records, restored_hashes, _ = recovery.restore_into_new_attempt(
                    source, target, state, restored, navigator,
                )
                self.assertEqual(opened.call_count, 32)
            self.assertEqual(len(restored.pil_frames), 41)
            self.assertFalse(navigator.frames)
            for index in range(10, 12):
                advance(restored)
                recovery.save_new_frames(target, restored.pil_frames, restored_hashes)
                release_evicted_payloads(restored, len(restored_hashes))
                restored_records.append({"action_index": index})
            self.assertEqual(len(restored_hashes), 49)
            self.assertEqual(sum(1 for _ in recovery.iter_saved_frames(target, restored_hashes)), 49)
            self.assertEqual(payload_summary(restored)["resident_counts"]["surfel_depths"], 32)
            self.assertEqual(len(list((source / "generated_frames").glob("*.png"))), 41)


if __name__ == "__main__":
    unittest.main()
