import ast
from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from generation_debug import GenerationDebug
from generation_rng import EXECUTION_DEFAULTS, PhaseRNG, RNG_SCHEMA, execution_settings, phase_seed
from scripts import run_vmem_demo_actions as runner
from scripts.build_vmem_transfer_manifest import build_rows
from scripts.run_vmem_demo_manifest import _load_manifest, _command_for_row
from scripts.vmem_protocol import expected_settings, verify_lock
from scripts.vmem_recovery import identity


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {"rng_mode": "isolated", "clip_attention": "math", "cut3r_attention": "math"}


class GenerationControlContractTest(unittest.TestCase):
    def test_v3_manifest_matches_builder_and_preserves_cases(self):
        actual = [{k: v for k, v in row.items() if k != "_manifest_line"}
                  for row in _load_manifest(ROOT / "manifests/vmem_transfer_v3.jsonl")]
        self.assertEqual(actual, build_rows("v3"))
        self.assertEqual(len(actual), 30)
        ignored = {"run_id", "_case_id", "checkpoint_every", *CONTROLS}
        for old, new in zip(build_rows("v2"), actual):
            self.assertEqual({k: v for k, v in old.items() if k not in ignored},
                             {k: v for k, v in new.items() if k not in ignored})
            self.assertEqual({k: new[k] for k in CONTROLS}, CONTROLS)
            self.assertEqual(new["checkpoint_every"], 5)
            self.assertEqual(expected_settings(new)["num_actions"], 195)
            self.assertNotIn("generation_debug", new)
            command = _command_for_row(new, output_root=Path("output"), config=Path("config"),
                                       device="cuda", dry_run=True)
            for key, value in CONTROLS.items():
                self.assertEqual(command[command.index("--" + key.replace("_", "-")) + 1], value)

    def test_production_controls_allow_long_checkpointed_runs_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "recovery").mkdir()
            (source / "recovery/latest.pt").touch()
            base = ["runner", "--image", "test_samples/oxford.jpg", "--duration-seconds", "60",
                    "--rng-mode", "isolated", "--clip-attention", "math", "--cut3r-attention", "math",
                    "--dry-run"]
            with patch.object(runner, "_load_runtime_dependencies") as load:
                for extra in ([], ["--resume-from", str(source), "--experiment-lock", "lock.json"]):
                    with patch("sys.argv", base + extra), patch("sys.stdout", new_callable=io.StringIO) as output:
                        runner.main()
                    record = json.loads(output.getvalue())
                    self.assertEqual(record["num_actions"], 195)
                    self.assertEqual(record["execution"], {**CONTROLS, "rng_schema": RNG_SCHEMA})
                with patch("sys.argv", base + ["--generation-debug", "isolated"]), self.assertRaises(ValueError):
                    runner.main()
                load.assert_not_called()

    def test_legacy_defaults_and_debug_execution_remain_explicit(self):
        self.assertEqual(execution_settings({}), {**EXECUTION_DEFAULTS, "rng_schema": "legacy"})
        self.assertEqual(execution_settings({"generation_debug": "isolated"})["rng_schema"], RNG_SCHEMA)
        for file in ("generation_rng.py", "generation_debug.py", "utils/util.py", "modeling/pipeline.py"):
            compile((ROOT / file).read_text(), file, "exec")
        image = ROOT / "test_samples/oxford.jpg"
        self.assertIn("generation_rng.py", runner._generation_provenance(image, ROOT / "configs/inference/inference.yaml")["source_sha256"])

    def test_lock_freezes_controls_and_schema_and_rejects_debug(self):
        row = {"image": "image", "run_id": "run", **CONTROLS, "frame_storage": "resident"}
        settings = expected_settings(row)
        provenance = {"source_sha256": {"file": "hash"}, "config_sha256": "config", "image_sha256": "image"}
        lock = {"schema": "vmem_experiment_lock_v4", "rows": [row], "image_sha256": {"image": "image"},
                "source_sha256": provenance["source_sha256"], "config_sha256": "config",
                "execution": {"run": execution_settings(row)}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            path.write_text(json.dumps(lock))
            verify_lock(path, settings, provenance)
            for key, value in {**EXECUTION_DEFAULTS, "generation_debug": "observe"}.items():
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_lock(path, {**settings, key: value}, provenance)
            lock["execution"]["run"]["rng_schema"] = "different"
            path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, "execution"):
                verify_lock(path, settings, provenance)
            lock["schema"] = "vmem_experiment_lock_v3"
            path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, "v4"):
                verify_lock(path, settings, provenance)

    def test_recovery_identity_includes_execution_controls_and_seed_schema(self):
        args = expected_settings({"image": "image", "run_id": "run", **CONTROLS})
        torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None))
        baseline = identity(args, {}, torch)
        self.assertEqual(baseline["execution"], {**CONTROLS, "rng_schema": RNG_SCHEMA})
        for key, value in EXECUTION_DEFAULTS.items():
            self.assertNotEqual(identity({**args, key: value}, {}, torch), baseline)

    def test_freeze_and_inventory_validate_execution_and_effective_config(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        self.addCleanup(sys.path.remove, str(ROOT / "scripts"))
        import audit_vmem_runs as audit
        try:
            import omegaconf
        except ImportError:
            self.skipTest("OmegaConf is required for effective-config validation")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, config, manifest, lock_path = [root / name for name in (
                "input.png", "config.yaml", "manifest.jsonl", "lock.json")]
            image.write_bytes(b"fixed input")
            config.write_text(json.dumps({"model": {"width": 576, "height": 576}, "surfel": {}, "inference": {}}))
            # Storage/trace validation has separate tests; isolate the execution contract.
            rows = [{**row, "image": str(image), "frame_storage": "legacy"} for row in build_rows("v3")[:2]]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            provenance = {"source_sha256": {"file": "hash"}, "config_sha256": audit.file_hash(config),
                          "image_sha256": audit.file_hash(image), "git_commit": "test"}
            args = SimpleNamespace(manifest=manifest, config=config, output=lock_path)
            with patch.object(audit, "_generation_provenance", return_value=provenance), patch("sys.stdout", new_callable=io.StringIO):
                audit.freeze(args)
                with self.assertRaises(FileExistsError):
                    audit.freeze(args)
            lock = json.loads(lock_path.read_text())
            self.assertEqual(lock["schema"], "vmem_experiment_lock_v4")
            self.assertEqual(lock["execution"][rows[0]["run_id"]], {**CONTROLS, "rng_schema": RNG_SCHEMA})
            settings = expected_settings(rows[0])
            provenance.update(checkpoint_sha256={"vmem": "weights", "cut3r": "weights"},
                              experiment_lock_sha256=verify_lock(lock_path, settings, provenance))
            path = root / "run"
            path.mkdir()
            execution = execution_settings(settings)
            spec = {"arguments": settings, "provenance": provenance, "execution": execution,
                    "torch_version": "test", "cuda_version": "test"}
            metadata = {**settings, "provenance": provenance, "execution": execution, "actual_frames": 781}
            effective = {"model": {"width": 576, "height": 576, "clip_attention": "math", "inference_num_steps": 50},
                         "surfel": {"cut3r_attention": "math", "niter": 400},
                         "inference": {"rng_mode": "isolated", "visualize": False}}
            for name, value in (("run_spec.json", spec), ("metadata.json", metadata),
                                ("run_status.json", {"status": "complete"}), ("actions.json", []),
                                ("retrieval_trace.json", []), ("generation_config.yaml", effective)):
                (path / name).write_text(json.dumps(value))
            with patch.object(audit, "validate_actions"), patch.object(audit, "read_resource_steps", return_value=[]), \
                    patch.object(audit, "validate_resources"), patch.object(audit, "validate_retrieval"), \
                    patch.object(audit, "probe_video", return_value={"frames": 781, "width": 576, "height": 576,
                                                                   "fps": 13, "seconds": 781 / 13}):
                self.assertEqual(audit.inspect_attempt(path, rows[0], lock_path, lock)["status"], "validated")
                metadata["execution"] = {**execution, "rng_schema": "changed"}
                (path / "metadata.json").write_text(json.dumps(metadata))
                self.assertEqual(audit.inspect_attempt(path, rows[0], lock_path, lock)["status"], "invalid")
                metadata["execution"] = execution
                (path / "metadata.json").write_text(json.dumps(metadata))
                effective["surfel"]["cut3r_attention"] = "native"
                (path / "generation_config.yaml").write_text(json.dumps(effective))
                self.assertEqual(audit.inspect_attempt(path, rows[0], lock_path, lock)["status"], "invalid")


class ProductionPhaseRNGTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is required")
        cls.torch = torch

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.control = PhaseRNG(seed=501, device="cpu", torch_module=self.torch)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(self.torch.set_rng_state, self.torch.get_rng_state())

    def test_exact_seed_rule_and_draws_match_original_debug_scheme(self):
        for name, step in (("model_load", -1), ("initialization", -1), ("diffusion", 9), ("reconstruction", 9)):
            payload = json.dumps(["vmem_phase_action_rng_v1", 501, name, step], separators=(",", ":"))
            seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2 ** 63)
            self.assertEqual(phase_seed(501, name, step), seed)
            expected_torch = self.torch.randn(8, generator=self.torch.Generator().manual_seed(seed))
            with self.control.phase(name, step):
                self.assertEqual(random.random(), random.Random(seed).random())
                np.testing.assert_array_equal(np.random.random(8), np.random.RandomState(seed % (2 ** 32)).random(8))
                self.assertTrue(self.torch.equal(self.torch.randn(8), expected_torch))

    def test_background_draws_and_failure_do_not_advance_next_diffusion(self):
        with self.control.phase("diffusion", 10):
            expected = self.torch.randn(8)
        self.torch.randn(100)
        before = self.torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with self.control.phase("reconstruction", 9):
                self.torch.randn(37 * 32)
                raise RuntimeError("failure")
        self.assertTrue(self.torch.equal(self.torch.get_rng_state(), before))
        with self.control.phase("reconstruction", 9):
            self.torch.randn(36 * 32)
        with self.control.phase("diffusion", 10):
            self.assertTrue(self.torch.equal(self.torch.randn(8), expected))

    def test_actual_sampling_function_has_same_draws_without_debug_hashes(self):
        torch = self.torch
        tree = ast.parse((ROOT / "utils/util.py").read_text())
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "do_sample")
        proxy = SimpleNamespace(inference_mode=torch.inference_mode, autocast=lambda *args: nullcontext(), randn=torch.randn)
        namespace = {"torch": proxy, "math": math, "nullcontext": nullcontext}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "utils/util.py", "exec"), namespace)
        debug = GenerationDebug(self.root / "debug.jsonl", seed=501, mode="isolated", device="cpu", torch_module=torch)
        arg = SimpleNamespace(to=lambda device: torch.zeros(1))
        common = dict(denoiser=None, model=None, ae=SimpleNamespace(decode=lambda value, _: value),
                      c={}, uc={}, c2w=arg, K=arg, cond_frames_mask=arg,
                      H=8, W=8, T=8, device="cpu", debug_step=9)

        def sample(_denoiser, noise, **kwargs):
            result = noise.clone()
            for step in range(3):
                draw = torch.randn_like(noise)
                if "debug_noise" in kwargs:
                    kwargs["debug_noise"](step, draw)
                result += draw
            return result

        # The diagnostic fingerprints conditioning too; keep its inputs serializable.
        fingerprint = debug.fingerprint
        debug.fingerprint = lambda value: None if value is arg else fingerprint(value)
        observed = namespace["do_sample"](sampler=sample, generation_debug=debug, **common)
        with patch.object(GenerationDebug, "fingerprint", side_effect=AssertionError("unexpected hashing")):
            production = namespace["do_sample"](sampler=sample, generation_control=self.control, **common)
        self.assertTrue(torch.equal(observed, production))
        self.assertEqual(list(self.root.iterdir()), [debug.path])


if __name__ == "__main__":
    unittest.main()
