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

    def _build_model(self) -> Sequential:
        """Builds and returns the Sequential CNN model configuration."""
        model = Sequential()
        # Block 1
        model.add(Conv2D(32, 3, activation='relu', padding='same',
                         input_shape=(self.image_size, self.image_size, self.img_channel)))
        model.add(Conv2D(32, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(padding='same'))
        model.add(Dropout(0.2))

        # Block 2
        model.add(Conv2D(64, 3, activation='relu', padding='same'))
        model.add(Conv2D(64, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(padding='same'))
        model.add(Dropout(0.3))

        # Block 3
        model.add(Conv2D(128, 3, activation='relu', padding='same'))
        model.add(Conv2D(128, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D(padding='same'))
        model.add(Dropout(0.4))

        # Fully connected layer
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))

        # Output layer
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def get_model(self):
        """Returns the compiled model."""
        return self.model