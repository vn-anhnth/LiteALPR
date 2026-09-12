import os
import sys
import unittest

import numpy as np

# Ensure repo root is on path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from litealpr import LiteALPR


class TestLiteALPRPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline on CPU for smoke test
        cls.alpr = LiteALPR(device="cpu")

    def test_pipeline_inference(self):
        # Create a synthetic plate image (white background with black dummy block)
        dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 255

        # Test detection output structure
        boxes = self.alpr.detect(dummy_img)
        self.assertIsInstance(boxes, list)

        # Test recognition output structure on a dummy crop
        dummy_crop = np.ones((32, 128, 3), dtype=np.uint8) * 255
        text, score = self.alpr.recognize(dummy_crop)
        self.assertIsInstance(text, str)
        self.assertIsInstance(score, float)

        # Test end-to-end read output structure
        results = self.alpr.read(dummy_img)
        self.assertIsInstance(results, list)

    def test_pytorch_weights_inference(self):
        from litealpr.pipeline import download_from_hf

        local_det_pt = os.path.join(
            root_dir, "output", "det", "yolov8n_efficient", "train", "weights", "best.pt"
        )
        local_rec_pth = os.path.join(root_dir, "output", "rec", "svtr26_tiny", "train", "best.pth")

        det_pt_path = (
            local_det_pt
            if os.path.exists(local_det_pt)
            else download_from_hf("yolov8n_efficient/best.pt")
        )
        rec_pth_path = (
            local_rec_pth
            if os.path.exists(local_rec_pth)
            else download_from_hf("svtr26_tiny/best.pth")
        )

        # Initialize pipeline using PyTorch weights (.pt and .pth)
        alpr_pt = LiteALPR(
            det_model_path=det_pt_path,
            rec_model_path=rec_pth_path,
            device="cpu",
        )

        dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        boxes = alpr_pt.detect(dummy_img)
        self.assertIsInstance(boxes, list)

        dummy_crop = np.ones((32, 128, 3), dtype=np.uint8) * 255
        text, score = alpr_pt.recognize(dummy_crop)
        self.assertIsInstance(text, str)
        self.assertIsInstance(score, float)

        results = alpr_pt.read(dummy_img)
        self.assertIsInstance(results, list)

    def test_explicit_onnx_weights_inference(self):
        from litealpr.pipeline import download_from_hf

        local_det_onnx = os.path.join(
            root_dir, "output", "det", "yolov8n_efficient", "train", "weights", "best_416.onnx"
        )
        local_rec_onnx = os.path.join(
            root_dir, "output", "rec", "svtr26_tiny", "train", "best.onnx"
        )

        det_onnx_path = (
            local_det_onnx
            if os.path.exists(local_det_onnx)
            else download_from_hf("yolov8n_efficient/best_416.onnx")
        )
        rec_onnx_path = (
            local_rec_onnx
            if os.path.exists(local_rec_onnx)
            else download_from_hf("svtr26_tiny/best.onnx")
        )

        # Initialize pipeline explicitly passing custom ONNX paths
        alpr_onnx = LiteALPR(
            det_model_path=det_onnx_path,
            rec_model_path=rec_onnx_path,
            device="cpu",
        )

        dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        boxes = alpr_onnx.detect(dummy_img)
        self.assertIsInstance(boxes, list)

        dummy_crop = np.ones((32, 128, 3), dtype=np.uint8) * 255
        text, score = alpr_onnx.recognize(dummy_crop)
        self.assertIsInstance(text, str)
        self.assertIsInstance(score, float)

        results = alpr_onnx.read(dummy_img)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
