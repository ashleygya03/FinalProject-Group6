"""Unit tests for model building logic."""
import unittest
from src.config import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES
from src.model import ASLModel


class TestModel(unittest.TestCase):
    def test_model_creation(self):
        """Test if the model is created with correct output shape."""
        asl_model = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
        model = asl_model.get_model()

        # Test input shape definition. The sequential model's first layer has this shape embedded.
        # Keras Sequential's input_shape incorporates the bath dimension (None).
        self.assertEqual(model.input_shape, (None, IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNEL))

        # Test output shape definition.
        self.assertEqual(model.output_shape, (None, N_CLASSES))


if __name__ == '__main__':
    unittest.main()