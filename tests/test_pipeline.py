import unittest
import numpy as np
import os
import sys

# Ensure repo root is on path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from litealpr import LiteALPR


class TestLiteALPRPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline on CPU for smoke test
        cls.alpr = LiteALPR(device='cpu')

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


if __name__ == '__main__':
    unittest.main()
