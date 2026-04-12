"""Model architecture class."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


class ASLModel:
    """Class to handle creation of the CNN model for ASL Classification."""

    def __init__(self, image_size: int, img_channel: int, n_classes: int):
        """Initialize parameters and build the model architecture."""
        self.image_size = image_size
        self.img_channel = img_channel
        self.n_classes = n_classes
        self.model = self._build_model()