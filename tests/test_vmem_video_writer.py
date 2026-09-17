import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from scripts import run_vmem_vbench_long as adapter


class VideoWriterHookTest(unittest.TestCase):
    def modules(self):
        vision = ModuleType("torchvision")
        vision.__version__ = "test"
        vision.io = ModuleType("torchvision.io")
        av = ModuleType("av")
        av.__version__ = "test"
        av.library_versions = {"libavcodec": (1, 2, 3)}
        return {"torchvision": vision, "torchvision.io": vision.io, "av": av}

    def test_supplies_missing_writer_and_is_idempotent(self):
        modules = self.modules()
        with patch.dict(sys.modules, modules):
            first = adapter.install_video_writer()
            second = adapter.install_video_writer()
        self.assertEqual(first, second)
        self.assertEqual(first["video_writer"], "vmem_pyav_video_only_v1")
        self.assertIs(modules["torchvision.io"].write_video, adapter.write_video_pyav)

    def test_keeps_native_writer(self):
        modules = self.modules()
        native = lambda *args, **kwargs: None
        modules["torchvision.io"].write_video = native
        with patch.dict(sys.modules, modules):
            runtime = adapter.install_video_writer()
        self.assertIs(modules["torchvision.io"].write_video, native)
        self.assertEqual(runtime["video_writer"], "torchvision.io.write_video")

    def test_missing_pyav_has_actionable_error(self):
        modules = self.modules()
        modules["av"] = None
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "vbench environment"):
            adapter.install_video_writer()

    def test_rejects_unsupported_rate_and_codec_without_loading_models(self):
        for rate in (0, -1, 12.5, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "integer fps"):
                adapter.write_video_pyav("unused.mp4", None, rate)
        with self.assertRaisesRegex(ValueError, "only libx264"):
            adapter.write_video_pyav("unused.mp4", None, 13, video_codec="mpeg4")


class VideoWriterEncodingTest(unittest.TestCase):
    def setUp(self):
        try:
            self.av = importlib.import_module("av")
            self.torch = importlib.import_module("torch")
            self.np = importlib.import_module("numpy")
        except ImportError:
            self.skipTest("CPU encoding tests require PyAV, PyTorch and NumPy; run --check in the VBench environment")

    def frames(self):
        frames = self.torch.zeros((26, 3, 16, 16), dtype=self.torch.uint8)
        frames[:, 0] = self.torch.arange(26, dtype=self.torch.uint8)[:, None, None] * 7 + 32
        frames[:, 1] = 96
        frames[:, 2] = 16
        return frames.permute(0, 2, 3, 1)

    def decode(self, path):
        with self.av.open(str(path)) as container:
            fps = float(container.streams.video[0].average_rate)
            frames = self.np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
        return fps, frames

    def test_noncontiguous_rgb_clip_preserves_shape_rate_and_order(self):
        source = self.frames()
        self.assertFalse(source.is_contiguous())
        original = source.clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            adapter.write_video_pyav(path, source, 13.0)
            fps, decoded = self.decode(path)
        self.assertEqual(fps, 13)
        self.assertEqual(decoded.shape, (26, 16, 16, 3))
        error = self.np.abs(decoded.astype(self.np.int16) - source.numpy().astype(self.np.int16))
        self.assertLessEqual(error.max(), 12)
        self.assertTrue(self.torch.equal(source, original))

    def test_rejects_wrong_layout_and_odd_dimensions(self):
        for shape in ((26, 3, 16, 16), (0, 16, 16, 3), (26, 15, 16, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                adapter.write_video_pyav("unused.mp4", self.torch.zeros(shape), 13)

    def test_matches_native_decoded_pixels_when_legacy_api_is_available(self):
        try:
            io = importlib.import_module("torchvision.io")
        except ImportError:
            self.skipTest("Native TorchVision comparison is unavailable")
        native = getattr(io, "write_video", None)
        if not callable(native) or native is adapter.write_video_pyav:
            self.skipTest("Native TorchVision write_video was removed")
        with tempfile.TemporaryDirectory() as tmp:
            reference, compatible = Path(tmp) / "native.mp4", Path(tmp) / "compatible.mp4"
            native(str(reference), self.frames(), fps=13)
            adapter.write_video_pyav(compatible, self.frames(), fps=13)
            reference_fps, reference_frames = self.decode(reference)
            fps, frames = self.decode(compatible)
        self.assertEqual(fps, reference_fps)
        self.np.testing.assert_array_equal(frames, reference_frames)


if __name__ == "__main__":
    unittest.main()
