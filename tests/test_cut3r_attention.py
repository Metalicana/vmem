import ast
from contextlib import nullcontext
import io
from pathlib import Path
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from clip_attention import attention_profile
from scripts.probe_vmem_reconstruction import attention_state


ROOT = Path(__file__).resolve().parents[1]
MATH_FLAGS = {"mha_fastpath": False, "flash_sdp": False, "memory_efficient_sdp": False,
              "math_sdp": True, "cudnn_sdp": False}


def load_function(relative, name, namespace, owner=None):
    tree = ast.parse((ROOT / relative).read_text())
    body = tree.body if owner is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return namespace[name]


class Cut3rAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("CPU PyTorch is required")
        cls.torch = torch

    def exercise_reconstruction(self, profile, failure=None):
        torch = self.torch
        before, rng = attention_state(torch), torch.get_rng_state().clone()
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
        expected = MATH_FLAGS if profile == "math" else before
        visited = []
        views = [{"idx": 0}, {"idx": 1}]
        predictions = [object(), object()]
        images, model, poses, depths = [object(), object()], object(), object(), object()
        result_values = tuple(object() for _ in range(5))

        def check(stage, flags):
            visited.append(stage)
            self.assertEqual(attention_state(torch), flags)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            if failure == stage:
                raise RuntimeError("injected failure")

        def prepare(**kwargs):
            check("prepare", before)
            self.assertIs(kwargs["pil_images"], images)
            return views

        def infer(actual_views, actual_model, device):
            check("inference", expected)
            self.assertIs(actual_views, views)
            self.assertIs(actual_model, model)
            self.assertEqual(device, "cpu")
            return {"views": views, "pred": predictions}, None

        def collate(value):
            check("collate", before)
            return value

        def align(output, actual_poses, actual_depths, lr, niter, outdir, device, save):
            check("alignment", before)
            self.assertIs(actual_poses, poses)
            self.assertIs(actual_depths, depths)
            self.assertEqual((lr, niter, device, save), (.01, 400, "cpu", False))
            self.assertIs(output["pred1"][0], predictions[0])
            self.assertIs(output["pred2"][0], predictions[1])
            return result_values

        namespace = {"torch": torch, "nullcontext": nullcontext, "time": time,
                     "prepare_input_from_pil": prepare, "collate_with_cat": collate, "prepare_output": align}
        function = load_function("extern/CUT3R/surfel_inference.py", "run_inference_from_pil", namespace)
        modules = {name: ModuleType(name) for name in ("src", "src.dust3r", "src.dust3r.inference")}
        modules["src.dust3r.inference"].inference = infer
        modules["src.dust3r.inference"].inference_recurrent = object()
        kwargs = {} if profile is None else {"inference_context": attention_profile(profile, torch)}
        try:
            with patch.dict("sys.modules", modules), patch("sys.stdout", new_callable=io.StringIO):
                with torch.no_grad(), torch.autocast("cpu"):
                    result = function(images, model, poses=poses, depths=depths, lr=.01, niter=400,
                                      device="cpu", visualize=False, **kwargs)
        finally:
            self.assertEqual(attention_state(torch), before)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(precision, (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()))
        self.assertEqual(visited, ["prepare", "inference"] + ["collate"] * 4 + ["alignment"])
        for key, value in zip(("point_clouds", "colors", "depths", "confidences", "camera_info"), result_values):
            self.assertIs(result[key], value)

    def test_real_reconstruction_function_scopes_math_to_inference(self):
        for profile in (None, "native", "math"):
            with self.subTest(profile=profile):
                self.exercise_reconstruction(profile)

    def test_real_reconstruction_function_restores_on_failures(self):
        for failure in ("inference", "collate", "alignment"):
            with self.subTest(failure=failure), self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.exercise_reconstruction("math", failure)

    def test_pipeline_passes_configured_profile_to_inference_boundary(self):
        torch = self.torch
        for profile in (None, "native", "math"):
            with self.subTest(profile=profile):
                before = attention_state(torch)
                expected = MATH_FLAGS if profile == "math" else before

                def inference(images, model, **kwargs):
                    self.assertEqual(attention_state(torch), before)
                    with kwargs["inference_context"]:
                        self.assertEqual(attention_state(torch), expected)
                    self.assertEqual(attention_state(torch), before)
                    raise RuntimeError("stop before surfel construction")

                namespace = {"np": np, "torch": torch, "attention_profile": attention_profile,
                             "run_inference_from_pil": inference}
                method = load_function("modeling/pipeline.py", "construct_and_store_scene", namespace,
                                       owner="VMemPipeline")
                config = SimpleNamespace(surfel={} if profile is None else {"cut3r_attention": profile},
                                         inference=SimpleNamespace(visualize_pointcloud=False))
                pipeline = SimpleNamespace(config=config, c2ws=[np.eye(4)], surfel_depths=[],
                                           surfel_model=object(), get_transformed_c2ws=lambda value: value)
                with self.assertRaisesRegex(RuntimeError, "stop before surfel construction"):
                    method(pipeline, [object()], time_indices=[0], input_time_indices=[0], device="cpu")
                self.assertEqual(attention_state(torch), before)


if __name__ == "__main__":
    unittest.main()
