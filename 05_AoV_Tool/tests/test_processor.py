
import unittest
import cv2
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()
        # Create a dummy 100x100 RGB image
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a white rectangle for edge detection
        cv2.rectangle(self.img, (20, 20), (80, 80), (255, 255, 255), -1)
        
    def test_gaussian_blur(self):
        """Test Gaussian Blur operation"""
        params = {"ksize": {"default": [5, 5]}, "sigmaX": {"default": 0}}
        result = self.processor._op_gaussian_blur(self.img, params, debug=False)
        self.assertEqual(result.shape, self.img.shape)
        self.assertNotEqual(np.sum(result), 0) # Should not be black

    def test_canny(self):
        """Test Canny Edge Detection"""
        params = {"threshold1": {"default": 50}, "threshold2": {"default": 150}}
        result = self.processor._op_canny(self.img, params, debug=False)
        # Canny returns BGR in this tool (ensure_bgr is called)
        self.assertEqual(result.shape, self.img.shape)
        # Edges should be detected around the rectangle
        self.assertTrue(np.mean(result) > 0)

    def test_grayscale_conversion(self):
        """Test implicit grayscale conversion helper"""
        gray = self.processor._ensure_gray(self.img)
        self.assertEqual(len(gray.shape), 2)
        
    def test_invalid_operation(self):
        """Test pipeline execution with unknown operation"""
        pipeline = [{"function": "NonExistentOp", "id": "node_0"}]
        # Should catch error/warning and return original/processed image without crash
        result = self.processor.execute_pipeline(self.img, pipeline, debug_mode=True)
        self.assertEqual(result.shape, self.img.shape)

    def test_integration_pipeline(self):
        """Test a mini pipeline: Blur -> Gray -> Canny"""
        pipeline = [
            {
                "function": "gaussian_blur",
                "parameters": {"ksize": {"default": [3, 3]}}
            },
            {
                "function": "bgr2gray",
                "parameters": {}
            },
            {
                "function": "canny_edge",
                "parameters": {"threshold1": {"default": 50}, "threshold2": {"default": 100}}
            }
        ]
        result = self.processor.execute_pipeline(self.img, pipeline)
        self.assertEqual(result.shape, self.img.shape) # Output is BGR even for Canny

if __name__ == '__main__':
    unittest.main()
