"""Unit tests for ResNet50 model building logic."""
import unittest
from src.config_resnet50 import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES
from src.model_resnet50 import ASLModel


class TestModel(unittest.TestCase):
    def test_model_creation(self):
        """Test if the ResNet50 model is created with correct input/output shapes."""
        asl_model = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
        model = asl_model.get_model()

        # Test input shape definition.
        self.assertEqual(model.input_shape, (None, IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNEL))

        # Test output shape definition.
        self.assertEqual(model.output_shape, (None, N_CLASSES))

    def test_base_model_is_resnet50(self):
        """Test that the base model is indeed ResNet50."""
        asl_model = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
        self.assertIn('resnet50', asl_model.base_model.name.lower())

    def test_base_model_frozen_by_default(self):
        """All ResNet50 layers should be frozen before fine-tuning."""
        asl_model = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
        trainable_layers = [l for l in asl_model.base_model.layers if l.trainable]
        self.assertEqual(len(trainable_layers), 0)


if __name__ == '__main__':
    unittest.main()
