import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_vmem_runs import _expand_trajectory_actions, inspect_attempt, validate_actions, validate_resources, validate_retrieval, validate_payload_snapshot
from vmem_protocol import commanded_path, expected_settings, scaling_actions, verify_lock

SPEC = importlib.util.spec_from_file_location("resource_audit", ROOT / "modeling" / "resource_audit.py")
RESOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOURCE)


def pipeline_fixture():
    return SimpleNamespace(
        pil_frames=[Image.new("RGB", (8, 8))], latents=[np.zeros((2, 4), dtype=np.float32)],
        surfels=[], surfel_to_timestep={}, memory_policy="slam_covisibility", memory_budget=32,
        memory_scope="surfel_indexed_view_memory", global_step=0,
        get_allowed_memory_indices=lambda: [0], _pinned_memory_frames=lambda: {0},
    )


class ResourceAccountingTest(unittest.TestCase):
    def test_resident_snapshot_waits_for_durable_output_and_release(self):
        pipeline = pipeline_fixture()
        pipeline.frame_storage = "resident"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            profiler = RESOURCE.ResourceProfiler(path)
            profiler.begin_step(pipeline)
            profiler.finish_step(pipeline)
            self.assertTrue(profiler.active)
            self.assertFalse(any(json.loads(line).get("event") == "step" for line in path.read_text().splitlines()))
            with self.assertRaises(RuntimeError):
                profiler.begin_step(pipeline)
            profiler.finish_step(pipeline, storage_ready=True)
            self.assertFalse(profiler.active)
            last = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(last["after_update_boundary"], "durable_output_and_payload_release")

    def test_payload_validator_rejects_retained_evictions_and_shared_storage(self):
        from frame_storage import payload_summary
        pipeline = pipeline_fixture()
        pipeline.frame_storage = "resident"
        pipeline.encoder_embeddings, pipeline.Ks = [np.ones(4)], [np.eye(3)]
        pipeline.surfel_depths = [np.ones((4, 4))]
        pipeline.durable_frame_count = 1
        snapshot = RESOURCE.memory_snapshot(pipeline)
        validate_payload_snapshot(snapshot, {0}, 1)
        shared = copy.deepcopy(snapshot)
        shared["frame_payloads"]["arrays_own_storage"] = False
        with self.assertRaises(ValueError):
            validate_payload_snapshot(shared, {0}, 1)
        pipeline.pil_frames.append(Image.new("RGB", (8, 8)))
        leaked = RESOURCE.memory_snapshot(pipeline)
        with self.assertRaises(ValueError):
            validate_payload_snapshot(leaked, {0}, 1)
        self.assertEqual(payload_summary(pipeline)["resident_counts"]["pil_frames"], 2)

    def test_numpy_slices_count_retained_backing_storage_once(self):
        root = np.zeros((8, 64), dtype=np.float32)
        first, second = root[:1], root[1:2]
        sizer = RESOURCE.StorageSizer()
        kept = sizer.component([first, second])
        self.assertEqual(kept["cpu_backing_bytes"], root.nbytes)
        alias = sizer.component([first])
        self.assertEqual(alias["cpu_backing_bytes"], 0)

    def test_pil_aliases_do_not_double_count_pixels(self):
        image = Image.new("RGB", (12, 10))
        pipeline = pipeline_fixture()
        pipeline.pil_frames = [image]
        snapshot = RESOURCE.memory_snapshot(pipeline, {"navigator_frames": [image]})
        self.assertEqual(snapshot["components"]["pil_frames"]["pil_pixel_bytes_estimate"], 360)
        self.assertEqual(snapshot["components"]["navigator_frames"]["pil_pixel_bytes_estimate"], 0)

    def test_tensor_numpy_aliases_share_one_backing_allocation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        array = np.zeros((4, 8), dtype=np.float32)
        tensor = torch.from_numpy(array)
        sizer = RESOURCE.StorageSizer(torch)
        self.assertEqual(sizer.component([array, tensor, tensor[1:].numpy()])["cpu_backing_bytes"], array.nbytes)

    def test_synchronized_stage_trace_and_warmup(self):
        syncs = []
        cuda = SimpleNamespace(synchronize=lambda device: syncs.append(device),
                               memory_allocated=lambda device: 10, memory_reserved=lambda device: 20,
                               max_memory_allocated=lambda device: 30, max_memory_reserved=lambda device: 40)
        fake_torch = SimpleNamespace(cuda=cuda, Tensor=type("FakeTensor", (), {}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            profiler = RESOURCE.ResourceProfiler(path, device="cuda:0", torch_module=fake_torch)
            pipeline = pipeline_fixture()
            profiler.begin_step(pipeline)
            for name in ("retrieval", "generation", "reconstruction", "memory_update"):
                profiler.begin_phase(name)
                profiler.end_phase(name)
            profiler.capture_before_update(pipeline)
            profiler.finish_step(pipeline)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(syncs), 8)
            self.assertTrue(events[-1]["warmup"])
            self.assertEqual(events[-1]["after_update"]["cuda_reserved_bytes"], 20)
            self.assertEqual(set(events[-1]["phase_seconds"]), {"retrieval", "generation", "reconstruction", "memory_update"})

    def test_failed_stage_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = RESOURCE.ResourceProfiler(Path(tmp) / "trace.jsonl")
            owner = pipeline_fixture()
            owner.resource_profiler = profiler
            profiler.begin_step(owner)

            @RESOURCE.profiled_phase("reconstruction")
            def failing(self):
                raise RuntimeError("synthetic failure")

            with self.assertRaises(RuntimeError):
                failing(owner)
            profiler.failure(RuntimeError("synthetic failure"))
            final = json.loads(profiler.path.read_text().splitlines()[-1])
            self.assertEqual(final["event"], "failure")
            self.assertEqual(final["phase"], "reconstruction")


class ProtocolValidationTest(unittest.TestCase):
    def test_paths_match_real_navigator_with_generation_stubbed(self):
        import scipy.spatial.transform as spt

        module = ast.parse((ROOT / "navigation.py").read_text())
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Navigator")
        names = {"_interpolate_poses", "move_forward", "move_backward", "turn_left", "turn_right", "_turn"}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"np": np, "spt": spt, "List": List, "Image": Image}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "navigation.py", "exec"), namespace)
        navigator_type = type("NavigatorUnderTest", (), {name: namespace[name] for name in names})
        for trajectory in ("fixed_region_v1", "expanding_excursions_v1", "pan_45", "pan_90", "out_and_back"):
            actions = _expand_trajectory_actions(SimpleNamespace(trajectory=trajectory, num_actions=585))
            navigator = navigator_type()
            navigator.current_pose, navigator.current_K = np.eye(4), np.eye(3)
            navigator.frames, navigator.pose_history = [], []
            navigator.retain_frame_history = False
            navigator.step_size, navigator.num_interpolation_frames = 0.02, 4
            navigator.pipeline = SimpleNamespace(generate_trajectory_frames=lambda *args, **kwargs:
                                                [Image.new("RGB", (2, 2)) for _ in range(4)])
            for action, planned in zip(actions, commanded_path(actions, 0.02)):
                if action in {"forward", "backward"}:
                    getattr(navigator, "move_" + action)()
                else:
                    method = navigator.turn_left if action.startswith("left") else navigator.turn_right
                    method(5 if action.endswith("5") else 10)
                np.testing.assert_allclose(navigator.current_pose, planned["current_pose"], atol=1e-6)
                self.assertEqual(navigator.frames, [])

    def test_scaling_prefixes_and_extent(self):
        for trajectory in ("fixed_region_v1", "expanding_excursions_v1"):
            short = scaling_actions(trajectory, 98)
            long = scaling_actions(trajectory, 585)
            self.assertEqual(short, long[:98])
            short_path = commanded_path(short, 0.02)
            long_path = commanded_path(long, 0.02)
            self.assertTrue(any(row["at_origin"] for row in short_path))
            if trajectory == "fixed_region_v1":
                self.assertAlmostEqual(long_path[-1]["max_radius"], 0.16)
            else:
                self.assertAlmostEqual(short_path[-1]["max_radius"], 0.48)
                self.assertAlmostEqual(long_path[-1]["max_radius"], 1.28)

    def test_camera_action_mismatch_is_rejected(self):
        settings = expected_settings({"run_id": "test", "image": "test.jpg", "trajectory": "fixed_region_v1", "num_actions": 16})
        planned = commanded_path(scaling_actions("fixed_region_v1", 16), settings["step_size"])
        actions = [{**row, "total_frames_after": row["frame_index"] + 1, "num_generated_frames": 4} for row in planned]
        validate_actions(actions, settings)
        actions[-1]["current_pose"][0][3] = 0.1
        with self.assertRaises(ValueError):
            validate_actions(actions, settings)

    def test_budget_includes_protected_frames(self):
        settings = expected_settings({"run_id": "test", "image": "test.jpg", "num_actions": 1,
                                      "memory_policy": "slam_covisibility", "memory_budget": 2})
        memory = {**RESOURCE.memory_snapshot(pipeline_fixture()),
                  "process_rss_bytes": 4096, "cuda_allocated_bytes": 10, "cuda_reserved_bytes": 20,
                  "cuda_peak_allocated_bytes": 30, "cuda_peak_reserved_bytes": 40,
                  "eligible_frame_indices": [0, 4], "eligible_frames": 2,
                  "protected_frame_indices": [0, 4], "protected_frames_eligible": True,
                  "surfel_references_within_eligible": True, "appearance_descriptor_sources": {"clip_image_encoder": 5}}
        steps = [{"step": 0, "frame_count": 5, "warmup": True, "policy": "slam_covisibility", "budget": 2,
                  "scope": settings["memory_scope"], "after_update": memory,
                  "before_update": copy.deepcopy(memory), "accounting_seconds": 0.01, "video_seconds": 5 / 13,
                  "phase_seconds": {key: 0.1 for key in ("generation", "retrieval", "reconstruction", "memory_update")}}]
        validate_resources(steps, settings)
        broken = copy.deepcopy(steps)
        broken[0]["after_update"]["eligible_frame_indices"].append(3)
        broken[0]["after_update"]["eligible_frames"] = 3
        with self.assertRaises(ValueError):
            validate_resources(broken, settings)
        missing_descriptor = copy.deepcopy(steps)
        missing_descriptor[0]["after_update"]["appearance_descriptor_sources"] = {}
        with self.assertRaises(ValueError):
            validate_resources(missing_descriptor, settings)
        missing_bytes = copy.deepcopy(steps)
        del missing_bytes[0]["after_update"]["components"]["surfel_depths"]
        with self.assertRaises(ValueError):
            validate_resources(missing_bytes, settings)
        missing_anchor = copy.deepcopy(steps)
        missing_anchor[0]["after_update"]["protected_frame_indices"] = [4]
        with self.assertRaises(ValueError):
            validate_resources(missing_anchor, settings)

    def test_retrieval_cannot_use_evicted_frame(self):
        trace = [{"global_step": 0, "allowed_memory_indices": [0], "selected_context_indices": [9], "target_frame_indices": [1, 2, 3, 4]}]
        steps = [{"frame_count": 5, "after_update": {"eligible_frame_indices": [0, 4]}}]
        with self.assertRaises(ValueError):
            validate_retrieval(trace, steps)

    def test_reconstruction_inputs_are_checked(self):
        trace = [{"global_step": 0, "allowed_memory_indices": [0], "selected_context_indices": [0], "target_frame_indices": [1, 2, 3, 4]}]
        steps = [{"frame_count": 5, "policy": "slam_covisibility", "scope": "surfel_indexed_view_memory",
                  "reconstruction_input_indices": [0, 1, 2, 3, 4], "after_update": {"eligible_frame_indices": [0, 4]}}]
        validate_retrieval(trace, steps)
        steps[0]["reconstruction_input_indices"] = [0, 4]
        with self.assertRaises(ValueError):
            validate_retrieval(trace, steps)

    def test_lock_and_unfinished_run_do_not_imply_completion(self):
        row = {"run_id": "test", "image": "test.jpg", "num_actions": 1}
        provenance = {"source_sha256": {"source": "abc"}, "config_sha256": "config", "image_sha256": "image"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / "lock.json"
            lock = {"source_sha256": provenance["source_sha256"], "config_sha256": "config",
                    "image_sha256": {"test.jpg": "image"}, "rows": [row]}
            lock_path.write_text(json.dumps(lock))
            settings = expected_settings(row)
            provenance["experiment_lock_sha256"] = verify_lock(lock_path, settings, provenance)
            with self.assertRaises(ValueError):
                verify_lock(lock_path, {**settings, "seed": 99}, provenance)
            run_dir = root / "test_run"
            run_dir.mkdir()
            (run_dir / "run_spec.json").write_text(json.dumps({"arguments": settings, "provenance": provenance}))
            (run_dir / "run_status.json").write_text(json.dumps({"status": "running"}))
            result = inspect_attempt(run_dir, {**row, "memory_policy": "unbounded"}, lock_path, lock)
            self.assertEqual(result["status"], "unfinished")
            self.assertTrue(result["provenance_verified"])


if __name__ == "__main__":
    unittest.main()
