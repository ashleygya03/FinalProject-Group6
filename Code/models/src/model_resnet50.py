"""Model architecture class for ASL classification using ResNet50."""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import ResNet50


class ASLModel:
    """Builds and manages the ResNet50-based ASL classifier."""

    def __init__(self, image_size: int, img_channel: int, n_classes: int):
        self.image_size = image_size
        self.img_channel = img_channel
        self.n_classes = n_classes
        self.base_model = None
        self.model = self._build_model()

    def _build_model(self):
        """Constructs the ResNet50 transfer-learning model.

        Phase 1 (frozen base): ResNet50 weights are locked; only the
        custom classifier head is trained.
        """
        self.base_model = ResNet50(
            input_shape=(self.image_size, self.image_size, self.img_channel),
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False  # freeze all ResNet50 layers

        inputs = Input(shape=(self.image_size, self.image_size, self.img_channel))
        x = self.base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model

    def get_model(self):
        """Returns the compiled-ready Keras model."""
        return self.model

    def unfreeze_top_layers(self, fine_tune_at=140):
        """Unfreezes the top ResNet50 layers for fine-tuning.

        Args:
            fine_tune_at: Layers before this index remain frozen.
                          ResNet50 has ~175 layers; unfreeze the last ~35
                          (matching the notebook's fine-tuning strategy).
        """
        self.base_model.trainable = True
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
