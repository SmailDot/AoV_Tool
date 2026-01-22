
import unittest
import numpy as np
import cv2
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processor import ImageProcessor

class TestRefactoredOps(unittest.TestCase):
    def setUp(self):
        self.processor = ImageProcessor()
        # Create dummy images
        self.img_color = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.img_color, (20, 20), (80, 80), (255, 255, 255), -1)
        self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        
        # Standard params
        self.params = {}

    def test_all_ops_smoke(self):
        """ Smoke test all registered operations """
        print("\n[Smoke Test] Testing all ops in operation_map...")
        
        failures = []
        
        for op_name, op_func in self.processor.operation_map.items():
            try:
                # Prepare inputs
                # Special handling for multi-input ops
                if op_name in ["add", "addWeighted", "bitwise_and", "bitwise_or", "bitwise_xor", "absdiff"]:
                    inputs = [self.img_color, self.img_color]
                    result = op_func(inputs, self.params, False)
                else:
                    result = op_func(self.img_color, self.params, False)
                
                if result is None:
                    failures.append(f"{op_name}: Returned None")
                elif not isinstance(result, np.ndarray):
                    failures.append(f"{op_name}: Returned {type(result)} instead of ndarray")
                else:
                    # Pass
                    pass
                    
            except Exception as e:
                failures.append(f"{op_name}: Crashed with {str(e)}")
        
        if failures:
            self.fail(f"The following ops failed smoke test:\n" + "\n".join(failures))
        else:
            print("[OK] All ops passed smoke test.")

if __name__ == '__main__':
    unittest.main()
