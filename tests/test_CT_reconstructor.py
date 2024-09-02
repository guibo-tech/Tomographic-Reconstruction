import unittest
import numpy as np
from project.CT_reconstructor import (
    load_image, perform_radon_transform, filtered_back_projection,
    fourier_transform, radon_and_iradon_reconstruction
)

class TestCTReconstructor(unittest.TestCase):

    def test_load_image(self):
        """Test if the image is loaded and normalized correctly."""
        image = load_image()
        self.assertIsInstance(image, np.ndarray, "The loaded image should be a numpy array.")
        self.assertEqual(image.dtype, np.float64, "The image should be normalized to floating point values.")
        self.assertLessEqual(image.max(), 1.0, "The maximum pixel value should be 1 after normalization.")
        self.assertGreaterEqual(image.min(), 0.0, "The minimum pixel value should be 0 after normalization.")

    def test_perform_radon_transform(self):
        """Test the Radon transform function."""
        image = load_image()
        thetas, rs, p, dtheta = perform_radon_transform(image)
        self.assertEqual(p.shape[0], len(rs), "The number of rows in p should match the length of rs.")
        self.assertEqual(p.shape[1], len(thetas), "The number of columns in p should match the length of thetas.")
        self.assertGreaterEqual(dtheta, 0, "dtheta should be a non-negative value.")

    def test_filtered_back_projection(self):
        """Test the filtered back projection function."""
        image = load_image()
        thetas, rs, p, dtheta = perform_radon_transform(image)
        fBP = filtered_back_projection(p, rs, thetas, dtheta)
        self.assertEqual(fBP.shape, image.shape, "The filtered back projection result should have the same shape as the original image.")
        self.assertIsInstance(fBP, np.ndarray, "The result should be a numpy array.")

    def test_fourier_transform(self):
        """Test the Fourier transform-based reconstruction function."""
        image = load_image()
        thetas, rs, p, dtheta = perform_radon_transform(image)
        f = fourier_transform(p, rs, thetas, dtheta)
        self.assertEqual(f.shape, image.shape, "The Fourier reconstruction result should have the same shape as the original image.")
        self.assertIsInstance(f, np.ndarray, "The result should be a numpy array.")

    def test_radon_and_iradon_reconstruction(self):
        """Test the Radon and iradon-based reconstruction function."""
        image = load_image()
        try:
            radon_and_iradon_reconstruction(image)
        except Exception as e:
            self.fail(f"radon_and_iradon_reconstruction raised an exception {e}")

if __name__ == '__main__':
    unittest.main()
