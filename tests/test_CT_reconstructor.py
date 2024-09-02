import unittest
import numpy as np
from project.CT_reconstructor import load_image


class TestCTReconstructor(unittest.TestCase):

    def test_load_image(self):
        """Test if the image is loaded and normalized correctly."""
        image = load_image()

        # Check if the result is a numpy array
        self.assertIsInstance(image, np.ndarray, "The loaded image should be a numpy array.")

        # Check if the image is normalized
        self.assertGreaterEqual(image.min(), 0.0, "The minimum pixel value should be 0 after normalization.")
        self.assertLessEqual(image.max(), 1.0, "The maximum pixel value should be 1 after normalization.")

        # Check if the image has the correct shape (2D)
        self.assertEqual(len(image.shape), 2, "The image should be a 2D array.")


if __name__ == '__main__':
    unittest.main()
